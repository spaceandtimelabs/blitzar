#pragma once

#include <utility>
#include <exception> // https://github.com/NVIDIA/cccl/issues/1278
#include "cub/cub.cuh"

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/base/digit_utility.h"
#include "sxt/multiexp/base/scalar_array.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// transform_to_digit 
//--------------------------------------------------------------------------------------------------
template <size_t BitWidth>
CUDA_CALLABLE uint8_t transform_to_digit(uint8_t byte, unsigned bucket_group_index) noexcept {
   static_assert(BitWidth == 8u || BitWidth == 4u);
   if constexpr (BitWidth == 8) {
     return byte;
   }
   if (bucket_group_index % 2 == 0) {
     return byte & 0xf; 
   } else {
     return byte >> 4u;
   }
}

//--------------------------------------------------------------------------------------------------
// bucket_sum_kernel 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T, size_t BitWidth>
__global__ void bucket_sum_kernel3(T* __restrict__ partial_sums, const T* __restrict__ generators,
                                   const uint8_t* __restrict__ scalars_t, unsigned n) noexcept {
  constexpr unsigned items_per_thread = 1;
  constexpr unsigned num_bucket_groups_per_byte = 8u / BitWidth;
  static_assert(8u % BitWidth == 0, "8 must be a multiple of BitWidth");

  auto thread_index = threadIdx.x;
  auto num_threads = blockDim.x;
  constexpr unsigned num_buckets_per_group = (1u << BitWidth) - 1u;
  constexpr unsigned element_num_bytes = 32u;

  auto bucket_group_index = blockIdx.x;
  auto byte_index = bucket_group_index / num_bucket_groups_per_byte;
  auto tile_index = blockIdx.y;
  auto output_index = blockIdx.z;

  auto num_bucket_groups = gridDim.x;
  auto num_tiles = gridDim.y;

  auto num_generators_per_tile = basn::divide_up(n, num_tiles);
  auto num_buckets_per_output = num_buckets_per_group * num_bucket_groups;
  auto num_partial_buckets_per_output = num_buckets_per_output * num_tiles;

  auto generator_first = num_generators_per_tile * tile_index;
  auto consumption_target = min(num_generators_per_tile, n - generator_first);
  auto generator_last = generator_first + consumption_target;

  // adjust the pointers
  scalars_t += n * byte_index + output_index * element_num_bytes * n;
  partial_sums += output_index * num_partial_buckets_per_output +
                  bucket_group_index * num_buckets_per_group * num_tiles;

  // set up a sum table initialized to the identity
  __shared__ T sums[num_buckets_per_group];
  for (unsigned i=thread_index; i<num_buckets_per_group; i+=num_threads) {
    sums[i] = T::identity();
  }

  // set up temp storage for algorithms
  using Sort = cub::BlockRadixSort<uint8_t, 32, items_per_thread, std::pair<unsigned, T>>;
  using Scan = cub::BlockScan<uint8_t, 32>;
  using Discontinuity = cub::BlockDiscontinuity<uint8_t, 32>;
  __shared__ union {
    Sort::TempStorage sort;
    Discontinuity::TempStorage discontinuity;
    Scan::TempStorage scan;
    uint8_t consumption_counter;
  } temp_storage;

  // sum buckets
  uint8_t digits[items_per_thread];
  std::pair<unsigned, T> gis[items_per_thread];
  uint8_t should_accumulate[items_per_thread];
  gis[0].first = generator_first + thread_index;
  if (gis[0].first < generator_last) {
    digits[0] = transform_to_digit<BitWidth>(scalars_t[gis[0].first], bucket_group_index);
    gis[0].second = generators[gis[0].first];
  } else {
    digits[0] = 0u;
  }
  generator_first += 32;

  unsigned num_generators_consumed = 0;
  while (num_generators_consumed < consumption_target) {
    // sort digit-g pairs
    Sort(temp_storage.sort).Sort(digits, gis);
    __syncthreads();

    // compute the difference between adjacent digit-g pairs
    Discontinuity(temp_storage.discontinuity)
        .FlagHeads(should_accumulate, digits, cub::Inequality());
    __syncthreads();
    should_accumulate[0] = should_accumulate[0] && digits[0] != 0u;

    // accumulate digits with no collisions
    if (should_accumulate[0]) {
      add_inplace(sums[digits[0]-1u], gis[0].second);
    }
    should_accumulate[0] = should_accumulate[0] || (digits[0] == 0u && gis[0].first < generator_last);

    // use a prefix sum on consumed generator indexes to update the index
    uint8_t offsets[items_per_thread];
    Scan(temp_storage.scan).InclusiveSum(should_accumulate, offsets);
    if (thread_index == num_threads - 1) {
      temp_storage.consumption_counter = offsets[0];
    }
    __syncthreads();
    gis[0].first = generator_first + offsets[0] - 1u;
    generator_first += temp_storage.consumption_counter;
    num_generators_consumed += temp_storage.consumption_counter;

    // if the generator was consumed, load a new element
    if (should_accumulate[0]) {
      if (gis[0].first < generator_last) {
        digits[0] = transform_to_digit<BitWidth>(scalars_t[gis[0].first], bucket_group_index);
        gis[0].second = generators[gis[0].first];
      } else {
        digits[0] = 0;
      }
    }
  }

  // copy partial sums to global memory
  for (unsigned sum_index = thread_index; sum_index < num_buckets_per_group;
       sum_index += num_threads) {
    partial_sums[sum_index * num_tiles + tile_index] = sums[sum_index];
  }
}

//--------------------------------------------------------------------------------------------------
// bucket_sum_combination_kernel
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
CUDA_CALLABLE void bucket_sum_combination_kernel3(T* __restrict__ sums, T* __restrict__ partial_sums,
                                                 unsigned num_tiles,
                                                 unsigned bucket_index) noexcept {
  partial_sums += bucket_index * num_tiles;
  T res = partial_sums[0];
  for (unsigned tile_index = 1; tile_index < num_tiles; ++tile_index) {
    auto e = partial_sums[tile_index];
    add_inplace(res, e);
  }
  sums[bucket_index] = res;
}

//--------------------------------------------------------------------------------------------------
// compute_bucket_sums
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> compute_bucket_sums3(basct::span<T> sums, basct::cspan<T> generators,
                                    basct::cspan<const uint8_t*> scalars,
                                    unsigned element_num_bytes, unsigned bit_width,
                                    unsigned max_num_tiles = 1u) noexcept {
  SXT_RELEASE_ASSERT(element_num_bytes == 32 && (bit_width == 8 || bit_width == 4),
                     "only support these values for now");
  auto n = static_cast<unsigned>(generators.size());
  auto num_outputs = static_cast<unsigned>(scalars.size());
  unsigned num_buckets_per_group = (1u << bit_width) - 1u;
  unsigned num_bucket_groups = basn::divide_up(32u * 8u, bit_width);
  auto num_buckets_per_output = num_buckets_per_group * num_bucket_groups;
  auto num_buckets = num_buckets_per_output * num_outputs;

  memmg::managed_array<uint8_t> scalars_t{n * element_num_bytes * num_outputs,
                                          memr::get_device_resource()};
  auto fut = mtxb::make_transposed_device_scalar_array(scalars_t, scalars, element_num_bytes, n);

  basdv::stream stream;
  memr::async_device_resource resource{stream};

  // copy generators to device
  memmg::managed_array<T> generators_dev{n, &resource};
  basdv::async_copy_host_to_device(generators_dev, generators, stream);

  // launch bucket accumulation kernel
  auto num_tiles = std::min(n, max_num_tiles);
  auto num_partial_sums = num_buckets * num_tiles;
  memmg::managed_array<T> partial_sums{num_partial_sums, &resource};
  co_await std::move(fut);
  /* std::print(stderr, "required shared memory: {}\n", num_buckets_per_group * sizeof(T)); */
  if (bit_width == 8u) {
    bucket_sum_kernel3<T, 8u><<<dim3(num_bucket_groups, num_tiles, num_outputs), 32, 0, stream>>>(
        partial_sums.data(), generators_dev.data(), scalars_t.data(), n);
  } else {
    bucket_sum_kernel3<T, 4u><<<dim3(num_bucket_groups, num_tiles, num_outputs), 32, 0, stream>>>(
        partial_sums.data(), generators_dev.data(), scalars_t.data(), n);
  }
  scalars_t.reset();
  generators_dev.reset();

  // combine the partial sums
  memmg::managed_array<T> sums_dev{sums.size(), &resource};
  auto combine = [
                     // clang-format off
    sums = sums_dev.data(),
    partial_sums = partial_sums.data(),
    num_tiles = num_tiles
                     // clang-format on
  ] __device__ __host__(unsigned /*num_buckets*/, unsigned bucket_index) noexcept {
    bucket_sum_combination_kernel3(sums, partial_sums, num_tiles, bucket_index);
  };
  algi::launch_for_each_kernel(stream, combine, num_buckets);
  basdv::async_copy_device_to_host(sums, sums_dev, stream);
  co_await xendv::await_stream(stream);
}
} // namespace sxt::mtxbk
