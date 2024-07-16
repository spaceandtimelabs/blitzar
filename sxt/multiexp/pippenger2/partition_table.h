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
#pragma once

#include <limits>

#include "sxt/base/bit/permutation.h"
#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/multiexp/pippenger2/constants.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// compute_partition_table_slice
//--------------------------------------------------------------------------------------------------
template <class U, bascrv::element T>
  requires requires(const U& u, const T& e) {
    static_cast<U>(e);
    T{u};
  }
CUDA_CALLABLE void compute_partition_table_slice(U* __restrict__ sums, unsigned window_width,
                                                 const T* __restrict__ generators) noexcept {
  assert(0u < window_width && window_width <= 32u);

  sums[0] = static_cast<U>(T::identity());

  // single entry sums
  for (unsigned i = 0; i < window_width; ++i) {
    sums[1u << i] = static_cast<U>(generators[i]);
  }

  // multi-entry sums
  for (unsigned k = 2; k <= window_width; ++k) {
    unsigned partition = std::numeric_limits<uint32_t>::max() >> (32u - k);
    auto partition_last = partition << (window_width - k);

    // iterate over all possible permutations with k bits set to 1
    // until we reach partition_last
    while (true) {
      // compute the k bit sum from a (k-1) bit sum and a 1 bit sum
      auto rest = partition & (partition - 1u);
      auto t = partition ^ rest;
      T sum{sums[rest]};
      T e{sums[t]};
      add_inplace(sum, e);
      sums[partition] = static_cast<U>(sum);
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
/**
 * Compute table of sums used for Pippenger's partition step with a given window width. Each
 * slice of the table contains all possible sums of a group of `window_width` generators.
 */
template <class U, bascrv::element T>
  requires requires(const U& u, const T& e) {
    static_cast<U>(e);
    T{u};
  }
void compute_partition_table(basct::span<U> sums, unsigned window_width,
                             basct::cspan<T> generators) noexcept {
  auto table_size = 1u << window_width;
  SXT_DEBUG_ASSERT(
      // clang-format off
     0u < window_width &&
     sums.size() == table_size * generators.size() / window_width &&
     generators.size() % window_width == 0
      // clang-format on
  );
  auto n = generators.size() / window_width;
  for (unsigned i = 0; i < n; ++i) {
    auto sums_slice = sums.subspan(i * table_size, table_size);
    auto generators_slice = generators.subspan(i * window_width, window_width);
    compute_partition_table_slice(sums_slice.data(), window_width, generators_slice.data());
  }
}

template <bascrv::element T>
void compute_partition_table(basct::span<T> sums, basct::cspan<T> generators) noexcept {
  compute_partition_table<T, T>(sums, 16u, generators);
}

template <bascrv::element T>
void compute_partition_table(basct::span<T> sums, unsigned window_width,
                             basct::cspan<T> generators) noexcept {
  compute_partition_table<T, T>(sums, window_width, generators);
}
} // namespace sxt::mtxpp2
