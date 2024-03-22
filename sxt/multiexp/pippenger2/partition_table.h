#pragma once

#include <limits>

#include "sxt/base/curve/element.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/num/choose.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// compute_partition_values
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
CUDA_CALLABLE void compute_partition_values(T* __restrict__ sums,
                                            const T* __restrict__ generators) noexcept {
  sums[0] = T::identity();

  // single entry sums
  for (unsigned i=0; i<16; ++i) {
    sums[1 << i] = generators[i];
  }

  // multi-entry sums
  for (unsigned k = 2; k <= 16; ++k) {
    auto partition = std::numeric_limits<uint16_t>::max() >> (16u - k);
    auto n = basn::choose_k(16u, k);
    for (unsigned i = 0; i < n; ++i) {
      auto rest = partition & (partition - uint16_t{1});
      auto t = partition ^ rest;
      auto sum = sums[rest];
      auto e = sums[t];
      add_inplace(sum, e);
      sums[partition] = sum;
    }
  }
}
} // namespace sxt::mtxpp2
