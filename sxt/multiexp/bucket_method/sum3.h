#pragma once

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
// bucket_sum_kernel 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T, size_t BitWidth>
__global__ void bucket_sum_kernel(T* __restrict__ partial_sums, const T* __restrict__ generators,
                                  const uint8_t* __restrict__ scalars_t, unsigned n) noexcept {
  constexpr size_t element_num_bytes = 32;
  constexpr size_t items_per_thread = 1;

  auto thread_index = threadIdx.x;
  auto num_threads = blockDim.x;
  constexpr unsigned num_buckets_per_group = (1u << BitWidth) - 1u;

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
  scalars_t += n * bucket_group_index + output_index * num_bucket_groups * n;
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
    should_accumulate[0] = max(should_accumulate[0], digits[0] == 0u);

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
// bucket_sum_combination_kernel
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
CUDA_CALLABLE void bucket_sum_combination_kernel(T* __restrict__ sums, T* __restrict__ partial_sums,
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
xena::future<> compute_bucket_sums(basct::span<T> sums, basct::cspan<T> generators,
                                   basct::cspan<const uint8_t*> scalars, unsigned element_num_bytes,
                                   unsigned bit_width) noexcept {
  SXT_RELEASE_ASSERT(element_num_bytes == 32 && bit_width == 8,
                     "only support these values for now");
  auto n = static_cast<unsigned>(generators.size());
  auto num_outputs = static_cast<unsigned>(scalars.size());
  unsigned num_buckets_per_group = (1u << bit_width) - 1u;
  unsigned num_bucket_groups = 32u;
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
  auto num_tiles = std::min(basn::divide_up(n, 32u), 4u);
  auto num_partial_sums = num_buckets * num_tiles;
  memmg::managed_array<T> partial_sums{num_partial_sums, &resource};
  co_await std::move(fut);
  bucket_sum_kernel<T, 8u><<<dim3(num_bucket_groups, num_tiles, num_outputs), 32, 0, stream>>>(
      partial_sums.data(), generators_dev.data(), scalars_t.data(), n);
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
    bucket_sum_combination_kernel(sums, partial_sums, num_tiles, bucket_index);
  };
  algi::launch_for_each_kernel(stream, combine, num_buckets);
  basdv::async_copy_device_to_host(sums, sums_dev, stream);
  co_await xendv::await_stream(stream);
}
} // namespace sxt::mtxbk
