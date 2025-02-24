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
public:
  explicit chunked_gpu_driver(double no_chunk_cutoff = 0.5) noexcept
      : no_chunk_cutoff_{no_chunk_cutoff} {}

  // driver
  xena::future<std::unique_ptr<workspace>>
  make_workspace(basct::cspan<T> mles,
                 basct::cspan<std::pair<T, unsigned>> product_table,
                 basct::cspan<unsigned> product_terms, unsigned n) const noexcept override {
    return {};
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
