#pragma once

#include "sxt/base/curve/element.h"
#include "sxt/base/macro/cuda_callable.h"

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
  }
  (void)sums;
  (void)generators;
}
} // namespace sxt::mtxpp2
