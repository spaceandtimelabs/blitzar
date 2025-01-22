/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "sxt/proof/sumcheck/gpu_driver.h"

#include <iostream>
#include <algorithm>
#include <chrono>
#include <print>

#include "sxt/base/device/synchronization.h"

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/algorithm/reduction/reduction.h"
#include "sxt/base/container/stack_array.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/panic.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/base/num/constexpr_switch.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/proof/sumcheck/constant.h"
#include "sxt/proof/sumcheck/partial_polynomial_mapper.h"
#include "sxt/proof/sumcheck/polynomial_mapper.h"
#include "sxt/proof/sumcheck/polynomial_utility.h"
#include "sxt/proof/sumcheck/sum_gpu.h"
#include "sxt/scalar25/operation/mul.h"
#include "sxt/scalar25/operation/muladd.h"
#include "sxt/scalar25/operation/sub.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// polynomial_reducer
//--------------------------------------------------------------------------------------------------
namespace {
template <unsigned MaxDegree> struct polynomial_reducer {
  using value_type = std::array<s25t::element, MaxDegree + 1u>;

  CUDA_CALLABLE static void accumulate_inplace(value_type& res, const value_type& e) noexcept {
    for (unsigned i = 0; i < res.size(); ++i) {
      s25o::add(res[i], res[i], e[i]);
    }
  }
};
} // namespace

//--------------------------------------------------------------------------------------------------
// gpu_workspace
//--------------------------------------------------------------------------------------------------
namespace {
struct gpu_workspace final : public workspace {
  memmg::managed_array<s25t::element> mles;
  memmg::managed_array<std::pair<s25t::element, unsigned>> product_table;
  memmg::managed_array<unsigned> product_terms;
  unsigned n;
  unsigned num_variables;

  gpu_workspace() noexcept
      : mles{memr::get_device_resource()}, product_table{memr::get_device_resource()},
        product_terms{memr::get_device_resource()} {}
};
} // namespace

//--------------------------------------------------------------------------------------------------
// make_workspace
//--------------------------------------------------------------------------------------------------
xena::future<std::unique_ptr<workspace>> gpu_driver::make_workspace(
    basct::cspan<s25t::element> mles,
    memmg::managed_array<std::pair<s25t::element, unsigned>>&& product_table_dev,
    memmg::managed_array<unsigned>&& product_terms_dev, unsigned n) const noexcept {
  auto ws = std::make_unique<gpu_workspace>();

  // dimensions
  ws->n = n;
  ws->num_variables = std::max(basn::ceil_log2(n), 1);

  // mles
  ws->mles = memmg::managed_array<s25t::element>{
      mles.size(),
      memr::get_device_resource(),
  };
  basdv::stream mle_stream;
  basdv::async_copy_host_to_device(ws->mles, mles, mle_stream);

  // product_table
  ws->product_table = std::move(product_table_dev);

  // product_terms
  ws->product_terms = std::move(product_terms_dev);

  // await
  co_await xendv::await_stream(mle_stream);
  co_return ws;
}

xena::future<std::unique_ptr<workspace>>
gpu_driver::make_workspace(basct::cspan<s25t::element> mles,
                           basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                           basct::cspan<unsigned> product_terms, unsigned n) const noexcept {
  auto ws = std::make_unique<gpu_workspace>();

  // dimensions
  ws->n = n;
  ws->num_variables = std::max(basn::ceil_log2(n), 1);

  // mles
  ws->mles = memmg::managed_array<s25t::element>{
      mles.size(),
      memr::get_device_resource(),
  };
  basdv::stream mle_stream;
  basdv::async_copy_host_to_device(ws->mles, mles, mle_stream);

  // product_table
  ws->product_table = memmg::managed_array<std::pair<s25t::element, unsigned>>{
      product_table.size(),
      memr::get_device_resource(),
  };
  basdv::stream product_table_stream;
  basdv::async_copy_host_to_device(ws->product_table, product_table, product_table_stream);

  // product_terms
  ws->product_terms = memmg::managed_array<unsigned>{
      product_terms.size(),
      memr::get_device_resource(),
  };
  basdv::stream product_terms_stream;
  basdv::async_copy_host_to_device(ws->product_terms, product_terms, product_terms_stream);

  // await
  co_await xendv::await_stream(mle_stream);
  co_await xendv::await_stream(product_table_stream);
  co_await xendv::await_stream(product_terms_stream);
  co_return ws;
}

//--------------------------------------------------------------------------------------------------
// sum
//--------------------------------------------------------------------------------------------------
xena::future<> gpu_driver::sum(basct::span<s25t::element> polynomial,
                               workspace& ws) const noexcept {
  auto& work = static_cast<gpu_workspace&>(ws);
  auto n = work.n;
  auto mid = 1u << (work.num_variables - 1u);
  SXT_RELEASE_ASSERT(
      // clang-format off
      work.n >= mid &&
      polynomial.size() - 1u <= max_degree_v
      // clang-format on
  );
  co_await sum_gpu(polynomial, work.mles, work.product_table, work.product_terms, n);
}

//--------------------------------------------------------------------------------------------------
// fold
//--------------------------------------------------------------------------------------------------
xena::future<> gpu_driver::fold(workspace& ws, const s25t::element& r) const noexcept {
  using s25t::operator""_s25;
  auto& work = static_cast<gpu_workspace&>(ws);
  auto n = work.n;
  auto mid = 1u << (work.num_variables - 1u);
  auto num_mles = work.mles.size() / n;
  SXT_RELEASE_ASSERT(
      // clang-format off
      work.n >= mid && work.mles.size() % n == 0
      // clang-format on
  );

  auto n1 = work.n - mid;
  s25t::element one_m_r = 0x1_s25;
  s25o::sub(one_m_r, one_m_r, r);

  memmg::managed_array<s25t::element> mles_p{num_mles * mid, memr::get_device_resource()};

  auto f = [
               // clang-format off
    mles_p = mles_p.data(),
    mles = work.mles.data(),
    n = n,
    num_mles = num_mles,
    r = r,
    one_m_r = one_m_r
               // clang-format on
  ] __device__
           __host__(unsigned mid, unsigned i) noexcept {
             for (unsigned mle_index = 0; mle_index < num_mles; ++mle_index) {
               auto val = mles[i + mle_index * n];
               s25o::mul(val, val, one_m_r);
               if (mid + i < n) {
                 s25o::muladd(val, r, mles[mid + i + mle_index * n], val);
               }
               mles_p[i + mle_index * mid] = val;
             }
           };
  auto fut = algi::for_each(f, mid);

  // complete
  co_await std::move(fut);

  {
    std::cerr << "******************************************\n";
    std::vector<s25t::element> mles_host(mles_p.size());
    basdv::memcpy_device_to_host(mles_host.data(), mles_p.data(), mles_p.size() * sizeof(s25t::element));
    for (auto& xi : mles_host) {
      std::cerr << "mle: " << xi << std::endl;
    }
  }

  work.n = mid;
  --work.num_variables;
  work.mles = std::move(mles_p);
}
} // namespace sxt::prfsk
