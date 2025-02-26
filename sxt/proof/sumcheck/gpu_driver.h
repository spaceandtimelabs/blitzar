/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2025-present Space and Time Labs, Inc.
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
#pragma once

#include <algorithm>

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/proof/sumcheck/driver.h"
#include "sxt/proof/sumcheck/sum_gpu.h"

namespace sxt::prfsk2 {
//--------------------------------------------------------------------------------------------------
// gpu_driver
//--------------------------------------------------------------------------------------------------
template <basfld::element T> class gpu_driver final : public driver<T> {
public:
  struct gpu_workspace final : public workspace {
    memmg::managed_array<T> mles;
    memmg::managed_array<std::pair<T, unsigned>> product_table;
    memmg::managed_array<unsigned> product_terms;
    unsigned n;
    unsigned num_variables;

    gpu_workspace() noexcept
        : mles{memr::get_device_resource()}, product_table{memr::get_device_resource()},
          product_terms{memr::get_device_resource()} {}
  };

  // driver
  xena::future<std::unique_ptr<workspace>>
  make_workspace(basct::cspan<T> mles, basct::cspan<std::pair<T, unsigned>> product_table,
                 basct::cspan<unsigned> product_terms, unsigned n) const noexcept override {
    auto ws = std::make_unique<gpu_workspace>();

    // dimensions
    ws->n = n;
    ws->num_variables = std::max(basn::ceil_log2(n), 1);

    // mles
    ws->mles = memmg::managed_array<T>{
        mles.size(),
        memr::get_device_resource(),
    };
    basdv::stream mle_stream;
    basdv::async_copy_host_to_device(ws->mles, mles, mle_stream);

    // product_table
    ws->product_table = memmg::managed_array<std::pair<T, unsigned>>{
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

  xena::future<> sum(basct::span<T> polynomial, workspace& ws) const noexcept override {
    auto& work = static_cast<gpu_workspace&>(ws);
    auto n = work.n;
    auto mid = 1u << (work.num_variables - 1u);
    SXT_RELEASE_ASSERT(
        // clang-format off
      work.n >= mid &&
      polynomial.size() - 1u <= max_degree_v
        // clang-format on
    );
    co_await sum_gpu<T>(polynomial, work.mles, work.product_table, work.product_terms, n);
  }

  xena::future<> fold(workspace& ws, const T& r) const noexcept override {
    auto& work = static_cast<gpu_workspace&>(ws);
    auto n = work.n;
    auto mid = 1u << (work.num_variables - 1u);
    auto num_mles = work.mles.size() / n;
    SXT_RELEASE_ASSERT(
        // clang-format off
      work.n >= mid && work.mles.size() % n == 0
        // clang-format on
    );

    T one_m_r = T::one();
    sub(one_m_r, one_m_r, r);

    memmg::managed_array<T> mles_p{num_mles * mid, memr::get_device_resource()};

    auto f =
        [
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
            mul(val, val, one_m_r);
            if (mid + i < n) {
              muladd(val, r, mles[mid + i + mle_index * n], val);
            }
            mles_p[i + mle_index * mid] = val;
          }
        };
    auto fut = algi::for_each(f, mid);

    // complete
    co_await std::move(fut);

    work.n = mid;
    --work.num_variables;
    work.mles = std::move(mles_p);
  }
};
} // namespace sxt::prfsk2
