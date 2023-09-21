#pragma once

#include <cstdint>
#include <concepts>

#include "sxt/base/num/divide_up.h"
#include "sxt/algorithm/base/mapper.h"
#include "sxt/algorithm/base/reducer.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// bucket_accumulate
//--------------------------------------------------------------------------------------------------
template <algb::reducer Reducer, unsigned int BlockSize, algb::mapper Mapper>
  requires std::same_as<typename Reducer::value_type, typename Mapper::value_type>
__global__ void bucket_accumulate(typename Reducer::value_type* bucket_sums,
                                  Mapper generator_mapper, const uint8_t* scalars,
                                  const unsigned* lengths) {
  using T = typename Reducer::value_type;
  auto block_index = blockIdx.x;
  auto scalar_byte_index = threadIdx.x;
  auto scalar_num_bytes = blockDim.x;
  auto num_blocks = gridDim.x;
  auto output_index = blockDim.y;
  auto num_buckets_per_generator = blockDim.x;
  static constexpr int bucket_size = 255;

  for (unsigned i=0; i<output_index; ++i) {
    scalars += *lengths++ * scalar_num_bytes;
  }

  auto num_generators = *lengths;
  auto num_generators_per_block = basn::divide_up(num_generators, num_blocks);
  auto generator_index = block_index * num_generators_per_block;
  auto generator_last = min(generator_index + num_generators_per_block, num_generators);

  bucket_sums += bucket_size * scalar_byte_index +
                 bucket_size * num_buckets_per_generator * block_index +
                 bucket_size * num_buckets_per_generator * num_blocks * output_index;

  scalars += scalar_byte_index + generator_index * scalar_num_bytes;

  for (int i=0; i<bucket_size; ++i) {
    bucket_sums[i] = Reducer::identity();
  }
  T g;
  for (; generator_index<generator_last; ++generator_index) {
    generator_mapper.map_index(g, generator_index);
    auto val = *scalars;
    scalars += scalar_num_bytes;
    if (val == 0) {
      continue;
    }
    Reducer::accumulate_inplace(bucket_sums[val-1], g);
  }
}
} // namespace sxt::mtxbk
