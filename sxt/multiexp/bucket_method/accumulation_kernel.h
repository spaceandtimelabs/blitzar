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

#include <concepts>
#include <cstdint>

#include "sxt/base/curve/element.h"
#include "sxt/base/num/divide_up.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// bucket_accumulate
//--------------------------------------------------------------------------------------------------
/**
 * Accumulate generators into buckets.
 *
 * This corresponds roughly to the 1st loop of Algorithm 1 described in
 *
 *    PipeMSM: Hardware Acceleration for Multi-Scalar Multiplication
 *    https://eprint.iacr.org/2022/999.pdf
 */
template <bascrv::element T>
__global__ void bucket_accumulate(T* bucket_sums, const T* generators, const uint8_t* scalars,
                                  unsigned length) {
  auto block_index = blockIdx.x;
  auto scalar_byte_index = threadIdx.x;
  auto scalar_num_bytes = blockDim.x;
  auto num_blocks = gridDim.x;
  auto output_index = blockIdx.y;
  auto num_bucket_groups = blockDim.x;
  static constexpr int bucket_group_size = 255;

  scalars += length * scalar_num_bytes * output_index;
  auto num_generators = length;
  auto num_generators_per_block = basn::divide_up(num_generators, num_blocks);
  auto generator_index = block_index * num_generators_per_block;
  auto generator_last = min(generator_index + num_generators_per_block, num_generators);

  bucket_sums += bucket_group_size * scalar_byte_index +
                 bucket_group_size * num_bucket_groups * block_index +
                 bucket_group_size * num_bucket_groups * num_blocks * output_index;

  scalars += scalar_byte_index + generator_index * scalar_num_bytes;

  auto identity = T::identity();
  for (int i = 0; i < bucket_group_size; ++i) {
    bucket_sums[i] = identity;
  }
  for (; generator_index < generator_last; ++generator_index) {
    auto val = *scalars;
    scalars += scalar_num_bytes;
    if (val == 0) {
      continue;
    }
    add(bucket_sums[val - 1], bucket_sums[val - 1], generators[generator_index]);
  }
}
} // namespace sxt::mtxbk
