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
template <basfld::element T>
class chunked_gpu_driver final : public driver<T> {
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
    return {};
  }

  xena::future<> fold(workspace& ws, const T& r) const noexcept override {
    return {};
  }

private:
  double no_chunk_cutoff_;
};
} // namespace sxt::prfsk2
