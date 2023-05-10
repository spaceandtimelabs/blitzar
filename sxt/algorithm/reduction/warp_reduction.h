/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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

#include <cassert>

#include "sxt/algorithm/base/reducer.h"

namespace sxt::algr {
//--------------------------------------------------------------------------------------------------
// warp_reduce
//--------------------------------------------------------------------------------------------------
/**
 * Efficiently reduce at most 32 elements in shared memory.
 *
 * Following nvidia's guidelines on efficient parallel reductions, we use
 * volatile for the shared memory which allows us to avoid calling __syncthreads().
 *
 * See https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf, page 22
 */
template <algb::reducer Reducer, unsigned int BlockSize>
__device__ void warp_reduce(volatile typename Reducer::value_type* shared_data,
                            unsigned int thread_index) {
  assert(thread_index < 32 && "can only be called on an individual warp");
  if constexpr (BlockSize >= 64) {
    Reducer::accumulate_inplace(shared_data[thread_index], shared_data[thread_index + 32]);
  }
  if constexpr (BlockSize >= 32) {
    if (thread_index < 16) {
      Reducer::accumulate_inplace(shared_data[thread_index], shared_data[thread_index + 16]);
    }
  }
  if constexpr (BlockSize >= 16) {
    if (thread_index < 8) {
      Reducer::accumulate_inplace(shared_data[thread_index], shared_data[thread_index + 8]);
    }
  }
  if constexpr (BlockSize >= 8) {
    if (thread_index < 4) {
      Reducer::accumulate_inplace(shared_data[thread_index], shared_data[thread_index + 4]);
    }
  }
  if constexpr (BlockSize >= 4) {
    if (thread_index < 2) {
      Reducer::accumulate_inplace(shared_data[thread_index], shared_data[thread_index + 2]);
    }
  }
  if constexpr (BlockSize >= 2) {
    if (thread_index < 1) {
      Reducer::accumulate_inplace(shared_data[thread_index], shared_data[thread_index + 1]);
    }
  }
}
} // namespace sxt::algr
