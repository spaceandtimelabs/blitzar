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

#include <algorithm>
#include <chrono>
#include <print>

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
#include "sxt/proof/sumcheck/constant.h"
#include "sxt/proof/sumcheck/partial_polynomial_mapper.h"
#include "sxt/proof/sumcheck/polynomial_mapper.h"
#include "sxt/proof/sumcheck/polynomial_utility.h"
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
  for (auto& pi : polynomial) {
    pi = {};
  }

  auto n1 = n - mid;

  // sum full terms
  xena::future<> fut1;
  auto f1 = [&]<unsigned MaxDegree>(std::integral_constant<unsigned, MaxDegree>) noexcept {
    if (n1 == 0) {
      fut1 = xena::make_ready_future();
      return;
    }
    polynomial_mapper<MaxDegree> mapper{
        .mles = work.mles.data(),
        .product_table = work.product_table.data(),
        .product_terms = work.product_terms.data(),
        .num_products = static_cast<unsigned>(work.product_table.size()),
        .mid = mid,
        .n = n,
    };
    auto fut = algr::reduce<polynomial_reducer<MaxDegree>>(basdv::stream{}, mapper, n1);
    fut1 = fut.then([&](std::array<s25t::element, MaxDegree + 1u> p) noexcept {
      for (unsigned i = 0; i < p.size(); ++i) {
        s25o::add(polynomial[i], polynomial[i], p[i]);
      }
    });
  };
  basn::constexpr_switch<1u, max_degree_v + 1u>(polynomial.size() - 1u, f1);

  // sum partial terms
  xena::future<> fut2;
  auto f2 = [&]<unsigned MaxDegree>(std::integral_constant<unsigned, MaxDegree>) noexcept {
    if (n1 == mid) {
      fut2 = xena::make_ready_future();
      return;
    }
    partial_polynomial_mapper<MaxDegree> mapper{
        .mles = work.mles.data() + n1,
        .product_table = work.product_table.data(),
        .product_terms = work.product_terms.data(),
        .num_products = static_cast<unsigned>(work.product_table.size()),
        .n = n,
    };
    auto fut = algr::reduce<polynomial_reducer<MaxDegree>>(basdv::stream{}, mapper, mid - n1);
    fut2 = fut.then([&](std::array<s25t::element, MaxDegree + 1u> p) noexcept {
      for (unsigned i = 0; i < p.size(); ++i) {
        s25o::add(polynomial[i], polynomial[i], p[i]);
      }
    });
  };
  basn::constexpr_switch<1u, max_degree_v + 1u>(polynomial.size() - 1u, f2);

  // await results
  co_await std::move(fut1);
  co_await std::move(fut2);
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

  // fold full terms
  auto f1 = [
                // clang-format off
    mles = work.mles.data(),
    n = n,
    num_mles = num_mles,
    mid = mid,
    r = r,
    one_m_r = one_m_r
                // clang-format on
  ] __device__
            __host__(unsigned /*n1*/, unsigned i) noexcept {
              for (unsigned mle_index = 0; mle_index < num_mles; ++mle_index) {
                auto data = mles + n * mle_index;
                auto val = data[i];
                s25o::mul(val, val, one_m_r);
                s25o::muladd(val, r, data[mid + i], val);
                data[i] = val;
              }
            };
  auto fut1 = algi::for_each(f1, n1);

  // fold partial terms
  auto f2 = [
                // clang-format off
    mles = work.mles.data() + n1,
    n = n,
    num_mles = num_mles,
    one_m_r = one_m_r
                // clang-format on
  ] __device__
            __host__(unsigned /*n1*/, unsigned i) noexcept {
              for (unsigned mle_index = 0; mle_index < num_mles; ++mle_index) {
                auto data = mles + n * mle_index;
                auto val = data[i];
                s25o::mul(val, val, one_m_r);
                data[i] = val;
              }
            };
  auto fut2 = algi::for_each(f2, mid - n1);

  // complete
  co_await std::move(fut1);
  co_await std::move(fut2);
  work.n = mid;
  --work.num_variables;
  work.mles.shrink(num_mles * mid);
}

xena::future<> gpu_driver::fold2(workspace& ws, const s25t::element& r) const noexcept {
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
             auto val = mles[i];
             s25o::mul(val, val, one_m_r);
             if (mid + i < n) {
               s25o::muladd(val, r, mles[mid + i], val);
             }
             mles_p[i] = val;
           };
  auto fut = algi::for_each(f, mid);

  // complete
  co_await std::move(fut);
  work.n = mid;
  --work.num_variables;
  work.mles = std::move(mles_p);
}
} // namespace sxt::prfsk
