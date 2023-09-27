#pragma once

#include <cstdint>
#include <concepts>

#include "sxt/base/curve/element.h"
#include "sxt/base/num/divide_up.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// bucket_accumulate
//--------------------------------------------------------------------------------------------------
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

  for (int i=0; i<bucket_group_size; ++i) {
    bucket_sums[i] = T::identity();
  }
  for (; generator_index<generator_last; ++generator_index) {
    auto val = *scalars;
    scalars += scalar_num_bytes;
    if (val == 0) {
      continue;
    }
    add(bucket_sums[val - 1], bucket_sums[val - 1], generators[generator_index]);
  }
}
} // namespace sxt::mtxbk
