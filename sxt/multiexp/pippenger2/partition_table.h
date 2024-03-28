#pragma once

#include <cassert>
#include <limits>
#include <print>

#include "sxt/base/bit/permutation.h"
#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/macro/cuda_callable.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// compute_partition_table_slice
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
CUDA_CALLABLE void compute_partition_table_slice(T* __restrict__ sums,
                                            const T* __restrict__ generators) noexcept {
  sums[0] = T::identity();

  // single entry sums
  for (unsigned i=0; i<16; ++i) {
    sums[1 << i] = generators[i];
  }

  // multi-entry sums
  for (unsigned k = 2; k <= 16; ++k) {
    unsigned partition = std::numeric_limits<uint16_t>::max() >> (16u - k);
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

//--------------------------------------------------------------------------------------------------
// compute_partition_table 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void compute_partition_table(basct::span<T> sums, basct::cspan<T> generators) noexcept {
  auto num_entries = 1u << 16u;
  SXT_DEBUG_ASSERT(
      // clang-format off
     sums.size() == num_entries * generators.size() / 16u &&
     generators.size() % 16 == 0
      // clang-format on
  );
  auto n = generators.size() / 16u;
  for (unsigned i=0; i<n; ++i) {
    auto sums_slice = sums.subspan(i * num_entries, num_entries);
    auto generators_slice = generators.subspan(i * 16u, 16u);
    compute_partition_table_slice<T>(sums_slice.data(), generators_slice.data());
  }
}
} // namespace sxt::mtxpp2
