#pragma once

#include "cub/cub.cuh"

#include "sxt/algorithm/block/runlength_count.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// multiproduct_table_kernel 
//--------------------------------------------------------------------------------------------------
template <uint16_t NumThreads, uint16_t ItemsPerThread, unsigned BitWidth>
static __global__ void multiproduct_table_kernel(uint16_t* __restrict__ bucket_counts,
                                                 uint16_t* __restrict__ indexes,
                                                 const uint8_t* __restrict__ bytes, unsigned n) {
  uint16_t thread_index = threadIdx.x;
  auto digit_index = blockDim.x;
  auto num_digits = gridDim.x;
  auto output_index = blockDim.y;
  auto num_buckets_per_digit = (1u << BitWidth) - 1u;

  // algorithms and shared memory
  using RadixSort = cub::BlockRadixSort<uint8_t, NumThreads, ItemsPerThread, uint16_t>;
  using RunlengthCount = algbk::runlength_count<uint8_t, uint16_t, NumThreads, (1u << BitWidth)>;
  __shared__ union {
    RadixSort::TempStorage sort;
    RunlengthCount::temp_storage count;
  } temp_storage;

  // adjust pointers
  bucket_counts += digit_index * num_buckets_per_digit;
  bucket_counts += output_index * num_digits * num_buckets_per_digit;
  indexes += digit_index * n;
  indexes += output_index * num_digits * n;
  bytes += digit_index * n;
  bytes += output_index * num_digits * n;

  // load bytes
  uint8_t keys[ItemsPerThread];
  uint16_t values[ItemsPerThread];
  for (uint16_t i=0; i<ItemsPerThread; ++i) {
    auto index = thread_index + i * NumThreads;
    if (index < n) {
      keys[i] = bytes[index];
      values[i] = index;
    } else {
      keys[i] = 0;
      values[i] = 0;
    }
  }

  // sort
  RadixSort(temp_storage.sort).Sort(keys, values);

  // count
  auto counts = RunlengthCount(temp_storage.count).count(keys);
  __syncthreads();

  // write counts
  for (unsigned i = thread_index; i < num_buckets_per_digit; i += NumThreads) {
    if (i > 0) {
      bucket_counts[i - 1] = counts[i];
    }
  }

  // write indexes
  auto zero_count = counts[0];
  for (unsigned i = 0; i < ItemsPerThread; ++i) {
    auto index = i + thread_index * ItemsPerThread;
    if (index >= zero_count) {
      indexes[i - zero_count] = values[i];
    }
  }
}
} // namespace sxt::mtxbk
