#pragma once

#include <cassert>
#include <limits>
#include <print>

#include "sxt/base/bit/permutation.h"
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
    auto partition = std::numeric_limits<uint16_t>::max() >> (16u - k);
    auto partition_last = partition << (16u - k);
    while (true) {
      auto rest = partition & (partition - 1u);
      auto t = partition ^ rest;
      auto sum = sums[rest];
      auto e = sums[t];
      add_inplace(sum, e);
      sums[partition] = sum;
      if (partition == partition_last) {
        break;
      }
      partition = basbt::next_permutation(partition);
    }
  }
}
} // namespace sxt::mtxpp2
