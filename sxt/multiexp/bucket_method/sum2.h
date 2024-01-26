#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/device_viewable.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/base/digit_utility.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// bucket_sum_kernel 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
__global__ void bucket_sum_kernel(T* __restrict__ partial_sums, const T* __restrict__ generators,
                                  const uint8_t* __restrict__ scalars, unsigned element_num_bytes,
                                  unsigned bit_width, unsigned n) noexcept {
  auto output_index = blockIdx.x;
  auto tile_index = blockIdx.y;
  auto num_tiles = gridDim.y;
  auto num_generators_per_tile = basn::divide_up(n, num_tiles);
  auto num_buckets_per_group = (1u << bit_width) - 1u;
  auto num_bucket_groups = blockDim.x;
  auto num_buckets_per_output = num_buckets_per_group * num_bucket_groups;
  auto num_partial_buckets_per_output = num_buckets_per_output * num_tiles;
  auto bucket_group_index = threadIdx.x;
  extern __shared__ T sum_array[];

  partial_sums += output_index * num_partial_buckets_per_output +
                  tile_index * num_buckets_per_output + bucket_group_index * num_buckets_per_group;
  scalars += output_index * n * element_num_bytes;

  // initialize the bucket partial sums
  T* __restrict__ sums = sum_array + num_buckets_per_group + bucket_group_index;
  for (unsigned sum_index = 0; sum_index < num_buckets_per_group; ++sum_index) {
    sums[sum_index] = T::identity();
  }

  // process
  auto generator_first = tile_index * num_generators_per_tile;
  auto generator_last = min(generator_first + num_generators_per_tile, n);

  __shared__ T e;
  __shared__ uint8_t scalar_data[32];
  basct::span<uint8_t> scalar{scalar_data, element_num_bytes};
  for (unsigned generator_index = generator_first; generator_index < generator_last;
       ++generator_index) {
    // load the generator into shared memory
    if (bucket_group_index == 0) {
      e = generators[generator_index];
    }

    // load the scalar into shared memory
    if (bucket_group_index < element_num_bytes) {
      scalar_data[bucket_group_index] =
          scalars[generator_index * element_num_bytes + bucket_group_index];
    }

    // extract digit and accumulate bucket sum
    uint8_t digit = 0;
    mtxb::extract_digit({&digit, 1u}, scalar, bit_width, bucket_group_index);
    if (digit != 0) {
      add(sums[digit], sums[digit], e);
    }
  }

  // copy partial sums to global memory
  for (unsigned sum_index = 0; sum_index < num_buckets_per_group; ++sum_index) {
    partial_sums[sum_index] = sums[sum_index];
  }
}

//--------------------------------------------------------------------------------------------------
// compute_bucket_sums2
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> compute_bucket_sums(basct::span<T> sums, basct::cspan<T> generators,
                                   basct::cspan<const uint8_t*> scalars, unsigned element_num_bytes,
                                   unsigned bit_width) noexcept {
  auto num_outputs = scalars.size();
  auto num_buckets_per_group = (1u << bit_width) - 1u;
  auto num_bucket_groups = basn::divide_up(element_num_bytes * 8u, bit_width);
  auto num_buckets_per_output = num_buckets_per_group * num_bucket_groups;

  auto shared_memory_num_bytes = num_buckets_per_output;
  auto num_tiles = 64u; // set better

  (void)sums;
  (void)generators;
  (void)scalars;
  (void)element_num_bytes;
  (void)bit_width;
  return {};
}
} // namespace sxt::mtxbk
