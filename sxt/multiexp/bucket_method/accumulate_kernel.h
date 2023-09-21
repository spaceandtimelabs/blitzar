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

  for (unsigned i=0; i<output_index; ++i) {
    scalars += *lengths++ * scalar_num_bytes;
  }

  auto num_generators = *lengths;
  auto num_generators_per_block = basn::divide_up(num_generators, num_blocks);
  auto generator_index = block_index * num_generators_per_block;
  auto generator_last = min(generator_index + num_generators_per_block, num_generators);

  T g;
  T sums[255];
  for (auto& elem : sums) {
    elem = T::identity();
  }
  for (; generator_index<generator_last; ++generator_index) {
    generator_mapper.map_index(g, generator_index);
    auto scalar = scalars + generator_index * scalar_num_bytes;
    auto val = scalar[scalar_byte_index];
    if (val == 0) {
      continue;
    }
    Reducer::accumulate_inplace(sums[val-1], g);
  }
  (void)num_blocks;
  (void)num_buckets_per_generator;

  (void)bucket_sums;
  (void)scalars;
  (void)num_generators;
}
} // namespace sxt::mtxbk
