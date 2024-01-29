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
  constexpr size_t num_buckets_per_group = (1u << BitWidth) - 1u;

  auto bucket_group_index = blockIdx.x;
  auto tile_index = blockIdx.y;
  auto output_index = blockIdx.z;

  auto num_bucket_groups = gridDim.x;
  auto num_tiles = gridDim.y;

  auto num_generators_per_tile = basn::divide_up(n, num_tiles);
  auto num_buckets_per_output = num_buckets_per_group * num_bucket_groups;
  auto num_partial_buckets_per_output = num_buckets_per_output * num_tiles;

  auto generator_first = num_generators_per_tile * tile_index;
  auto generator_last = min(generator_first + num_generators_per_tile, n);

  // adjust the pointers
  scalars_t += n * bucket_group_index;
  partial_sums += output_index * num_partial_buckets_per_output +
                  bucket_group_index * num_buckets_per_group * num_tiles;

  // set up a sum table initialized to the identity
  __shared__ T sums[num_buckets_per_group];
  for (unsigned i=thread_index; i<num_buckets_per_group; i+=num_threads) {
    sums[i] = T::identity();
  }

  // set up temp storage for algorithms
  using Sort = cub::BlockRadixSort<uint8_t, 32, items_per_thread>;
  using Scan = cub::BlockScan<uint8_t, 32>;
  using Discontinuity = cub::BlockDiscontinuity<uint8_t, 32>;
  using Reduce = cub::BlockReduce<uint8_t, 32>;
  __shared__ union {
    Sort::TempStorage sort;
    Discontinuity::TempStorage discontinuity;
    Scan::TempStorage scan;
    Reduce::TempStorage reduce;
  } temp_storage;

  // sum buckets
  unsigned index = generator_first + thread_index;
  uint8_t digits[items_per_thread];
  T gs[items_per_thread];
  uint8_t should_accumulate[items_per_thread];
  digits[0] = scalars_t[index];
  gs[0] = generators[index];
  unsigned index_p = generator_first + num_threads;
  while (true) {
    // sort digit-g pairs
    Sort(temp_storage.sort).Sort(digits, gs);

    // compute the difference between adjacent digit-g pairs
    Discontinuity(temp_storage.discontinuity)
        .FlagHeads(should_accumulate, digits, cub::Inequality());

    // accumulate digits with no collisions
    if (should_accumulate[0] && digits[0] != 0u) {
      add_inplace(sums[digits[0]-1u], gs[0]);
    }
    should_accumulate[0] *= (digits[0] == 0u);

    // use a prefix sum on consumed generator indexes to update the index
    uint8_t offsets[items_per_thread];
    Scan(temp_storage.scan).InclusiveSum(should_accumulate, offsets);
    auto num_consumed = Reduce(temp_storage.reduce).Sum(should_accumulate);
                   // Note: probably a more efficient way to do this using the
                   // prefix sums
    index = index_p + offsets[0];
    index_p += num_consumed;
    if (index >= generator_last) {
      return;
    }

    // if the generator was consumed, load a new element
    if (should_accumulate[0]) {
      digits[0] = scalars_t[index];
      gs[0] = generators[index];
    }
  }

  // copy partial sums to global memory
  for (unsigned sum_index = 0; sum_index < num_buckets_per_group; ++sum_index) {
    partial_sums[sum_index * num_tiles + tile_index] = sums[sum_index];
  }
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
