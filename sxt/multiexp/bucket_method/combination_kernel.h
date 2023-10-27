/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
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

#include "sxt/base/curve/element.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/base/num/log2p1.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// combine_partial_bucket_sums
//--------------------------------------------------------------------------------------------------
/**
 * Suppose we have a multi-dimensional array of partial bucket sums
 *
 *     Bp[bucket_index, bucket_group_index, output_index, partial_sum_index]
 *
 * Then this kernel combines all of the partial sums to produce the result
 *
 *     B[bucket_index, bucket_group_index, output_index] = 
 *                sum_{partial_sum_index} Bp[bucket_index, bucket_group_index, output_index]
 */
template <bascrv::element T>
__global__ void combine_partial_bucket_sums(T* out, T* partial_bucket_sums, unsigned num_partials) {
  assert(num_partials > 0);

  auto bucket_group_size = gridDim.x;
  auto bucket_group_index = threadIdx.x;
  auto num_bucket_groups = blockDim.x;
  auto bucket_index = blockIdx.x;
  auto output_index = blockIdx.y;

  partial_bucket_sums += bucket_index + bucket_group_size * bucket_group_index +
                         bucket_group_size * num_bucket_groups * num_partials * output_index;

  out += bucket_index + bucket_group_size * bucket_group_index +
         bucket_group_size * num_bucket_groups * output_index;

  T sum = *partial_bucket_sums;
  partial_bucket_sums += bucket_group_size * num_bucket_groups;
  for (unsigned i = 1; i < num_partials; ++i) {
    add_inplace(sum, *partial_bucket_sums);
    partial_bucket_sums += bucket_group_size * num_bucket_groups;
  }

  *out = sum;
}

//--------------------------------------------------------------------------------------------------
// combine_bucket_groups
//--------------------------------------------------------------------------------------------------
/**
 * Suppose we have a multi-dimensional array of grouped buckets
 *   
 *    B[bucket_index, bucket_group_index, output_index]
 *
 * This kernel combines the bucket groups to produce
 *
 *    B'[bucket_index, output_index] = 
 *            sum_{bucket_group_index} 
*                    2^{log2(BucketGroupSize+1) * bucket_group_index} * B[bucket_index, bucket_group_index, output_index]
 */
template <unsigned BucketGroupSize, unsigned NumBucketGroups, bascrv::element T>
__global__ void combine_bucket_groups(T* out, T* bucket_sums) {
  assert(NumBucketGroups > 0);
  auto thread_index = threadIdx.x;
  auto block_index = blockIdx.x;
  auto num_threads = blockDim.x;
  auto bucket_index = thread_index + block_index * num_threads;
  if (bucket_index >= BucketGroupSize) {
    return;
  }

  auto output_index = blockIdx.y;

  bucket_sums += bucket_index + BucketGroupSize * NumBucketGroups * output_index;
  out += bucket_index + BucketGroupSize * output_index;

  unsigned i = NumBucketGroups - 1;
  T sum = bucket_sums[i * BucketGroupSize];
  constexpr auto bucket_group_size_log2p1 = basn::log2p1(BucketGroupSize);
  while (i-- > 0) {
    for (int j = 0; j < bucket_group_size_log2p1; ++j) {
      double_element(sum, sum);
    }
    add_inplace(sum, bucket_sums[BucketGroupSize * i]);
  }

  *out = sum;
}
} // namespace sxt::mtxbk
