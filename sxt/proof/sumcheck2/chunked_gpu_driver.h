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

#include "sxt/algorithm/iteration/transform.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/proof/sumcheck2/device_cache.h"
#include "sxt/proof/sumcheck2/driver.h"
#include "sxt/proof/sumcheck2/fold_gpu.h"
#include "sxt/proof/sumcheck2/gpu_driver.h"
#include "sxt/proof/sumcheck2/mle_utility.h"
#include "sxt/proof/sumcheck2/sum_gpu.h"

namespace sxt::prfsk2 {
//--------------------------------------------------------------------------------------------------
// chunked_gpu_driver
//--------------------------------------------------------------------------------------------------
template <basfld::element T> class chunked_gpu_driver final : public driver<T> {
  struct chunked_gpu_workspace final : public workspace {
    std::unique_ptr<workspace> single_gpu_workspace;

    device_cache<T> cache;
    memmg::managed_array<T> mles_data;
    basct::cspan<T> mles;
    unsigned n;
    unsigned num_variables;

    chunked_gpu_workspace(basct::cspan<std::pair<T, unsigned>> product_table,
                          basct::cspan<unsigned> product_terms) noexcept
        : cache{product_table, product_terms} {}
  };

  static xena::future<> try_make_single_gpu_workspace(chunked_gpu_workspace& work,
                                                      double no_chunk_cutoff) noexcept {
    auto gpu_memory_fraction = get_gpu_memory_fraction<T>(work.mles);
    if (gpu_memory_fraction > no_chunk_cutoff) {
      co_return;
    }

    // construct single gpu workspace
    auto cache_data = work.cache.clear();
    gpu_driver<T> drv;
    work.single_gpu_workspace =
        co_await drv.make_workspace(work.mles, std::move(cache_data->product_table),
                                    std::move(cache_data->product_terms), work.n);

    // free data we no longer need
    work.mles_data.reset();
    work.mles = {};
  }

public:
  explicit chunked_gpu_driver(double no_chunk_cutoff = 0.5) noexcept
      : no_chunk_cutoff_{no_chunk_cutoff} {}

  // driver
  xena::future<std::unique_ptr<workspace>>
  make_workspace(basct::cspan<T> mles, basct::cspan<std::pair<T, unsigned>> product_table,
                 basct::cspan<unsigned> product_terms, unsigned n) const noexcept override {
    auto res = std::make_unique<chunked_gpu_workspace>(product_table, product_terms);
    res->mles = mles;
    res->n = n;
    res->num_variables = std::max(basn::ceil_log2(n), 1);
    auto gpu_memory_fraction = get_gpu_memory_fraction(mles);
    if (gpu_memory_fraction <= no_chunk_cutoff_) {
      gpu_driver<T> drv;
      res->single_gpu_workspace =
          co_await drv.make_workspace(mles, product_table, product_terms, n);
    }
    co_return std::unique_ptr<workspace>(std::move(res));
  }

  xena::future<> sum(basct::span<T> polynomial, workspace& ws) const noexcept override {
    auto& work = static_cast<chunked_gpu_workspace&>(ws);
    if (work.single_gpu_workspace) {
      gpu_driver<T> drv;
      co_return co_await drv.sum(polynomial, *work.single_gpu_workspace);
    }
    co_await sum_gpu<T>(polynomial, work.cache, work.mles, work.n);
  }

  xena::future<> fold(workspace& ws, const T& r) const noexcept override {
    auto& work = static_cast<chunked_gpu_workspace&>(ws);
    if (work.single_gpu_workspace) {
      gpu_driver<T> drv;
      co_return co_await drv.fold(*work.single_gpu_workspace, r);
    }
    auto n = work.n;
    auto mid = 1u << (work.num_variables - 1u);
    auto num_mles = work.mles.size() / n;
    SXT_RELEASE_ASSERT(
        // clang-format off
      work.n >= mid && work.mles.size() % n == 0
        // clang-format on
    );

    auto one_m_r = T::one();
    sub(one_m_r, one_m_r, r);

    // fold
    memmg::managed_array<T> mles_p(num_mles * mid);
    co_await fold_gpu<T>(mles_p, work.mles, n, r);

    // update
    work.n = mid;
    --work.num_variables;
    work.mles_data = std::move(mles_p);
    work.mles = work.mles_data;

    // check if we should fall back to single gpu workspace
    co_await try_make_single_gpu_workspace(work, no_chunk_cutoff_);
  }

private:
  double no_chunk_cutoff_;
};
} // namespace sxt::prfsk2
