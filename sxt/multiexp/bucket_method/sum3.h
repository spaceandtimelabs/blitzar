#pragma once

#include <exception> // https://github.com/NVIDIA/cccl/issues/1278
#include "cub/cub.cuh"

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/device/device_viewable.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/base/digit_utility.h"
#include "sxt/multiexp/base/scalar_array.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// bucket_sum_kernel 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T, size_t BitWidth>
__global__ void bucket_sum_kernel(T* __restrict__ partial_sums, const T* __restrict__ generators,
                                  const uint8_t* __restrict__ scalars_t, unsigned element_num_bytes,
                                  unsigned n) noexcept {
  constexpr size_t items_per_thread = 1;

  auto thread_index = threadIdx.x;
  auto num_threads = blockDim.x;
  constexpr size_t num_buckets_per_output = (1u << BitWidth) - 1u;
  auto digit_index = blockIdx.x;
  auto tile_index = blockIdx.y;
  auto num_tiles = gridDim.y;

  auto num_generators_per_tile = basn::divide_up(n, num_tiles);
  auto generator_first = num_generators_per_tile * tile_index;
  auto generator_last = min(generator_first + num_generators_per_tile, n);

  scalars_t += n * digit_index;

  // set up a sum table initialized to the identity
  __shared__ T sums[num_buckets_per_output];
  for (unsigned i=thread_index; i<num_buckets_per_output; i+=num_threads) {
    sums[i] = T::identity();
  }

  // sum buckets
  using Sort = cub::BlockRadixSort<uint8_t, 32, items_per_thread>;
  __shared__ union {
    Sort::TempStorage sort;
  } temp_storage;

  unsigned index = generator_first + thread_index;
  uint8_t digits[items_per_thread];
  T gs[items_per_thread];
  digits[0] = scalars_t[index];
  gs[0] = generators[index];
  while (index < generator_last) {
    // sort digit-g pairs
    Sort(temp_storage.sort).Sort(digits, gs);

    // compute the difference between adjacent digit-g pairs

    // is this digit free of collisions
    // if so
    //      add_inplace(sums[digit-1u], g)

    // use a prefix sum on consumed generator indexes to update the index

    // if the generator was consumed, load a new element
  }
  (void)sums;
  (void)num_threads;
  (void)partial_sums;
  (void)generators;
  (void)element_num_bytes;
  (void)n;
}

//--------------------------------------------------------------------------------------------------
// compute_bucket_sums
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void compute_bucket_sums(basct::span<T> sums, const basdv::stream& stream,
                         basct::cspan<T> generators, basct::cspan<const uint8_t*> scalars,
                         unsigned element_num_bytes, unsigned bit_width) noexcept {
  (void)sums;
  (void)stream;
  (void)generators;
  (void)scalars;
  (void)element_num_bytes;
  (void)bit_width;
}
} // namespace sxt::mtxbk
