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

#include "cub/cub.cuh"
#include "sxt/base/macro/cuda_callable.h"

namespace sxt::algbk {
//--------------------------------------------------------------------------------------------------
// runlength_count
//--------------------------------------------------------------------------------------------------
/**
 * This is adapted from block_histogram_sort in CUB. See
 *   https://github.com/NVIDIA/cccl/blob/a51b1f8c75f8e577eeccc74b45f1ff16a2727265/cub/cub/block/specializations/block_histogram_sort.cuh
 */
template <class T, class CounterT, unsigned NumThreads, unsigned NumBins> class runlength_count {
  using Discontinuity = cub::BlockDiscontinuity<T, NumThreads>;

public:
  struct temp_storage {
    typename Discontinuity::TempStorage discontinuity;
    CounterT run_begin[NumBins];
    CounterT run_end[NumBins];
  };

  CUDA_CALLABLE explicit runlength_count(temp_storage& storage) noexcept
      : storage_{storage}, discontinuity_{storage.discontinuity} {}

  /**
   * If items holds sorted items across threads in a block, count and return a
   * pointer to a table of the items' run lengths.
   */
  template <unsigned ItemsPerThread>
  CUDA_CALLABLE CounterT* count(T (&items)[ItemsPerThread]) noexcept {
    auto thread_id = threadIdx.x;
    for (unsigned i = thread_id; i < NumBins; i += NumThreads) {
      storage_.run_begin[i] = NumThreads * ItemsPerThread;
      storage_.run_end[i] = NumThreads * ItemsPerThread;
    }
    int flags[ItemsPerThread];
    auto flag_op = [&storage = storage_](T a, T b, int b_index) noexcept {
      if (a != b) {
        storage.run_begin[b] = static_cast<CounterT>(b_index);
        storage.run_end[a] = static_cast<CounterT>(b_index);
        return true;
      } else {
        return false;
      }
    };
    __syncthreads();
    discontinuity_.FlagHeads(flags, items, flag_op);
    if (thread_id == 0) {
      storage_.run_begin[items[0]] = 0;
    }
    __syncthreads();
    for (unsigned i = thread_id; i < NumBins; i += NumThreads) {
      storage_.run_end[i] -= storage_.run_begin[i];
    }
    return storage_.run_end;
  }

private:
  temp_storage& storage_;
  Discontinuity discontinuity_;
};
} // namespace sxt::algbk
