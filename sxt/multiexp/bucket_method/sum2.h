#pragma once

#include <print>

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/device_viewable.h"
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
  extern __shared__ std::byte sum_array_data[];
  auto sum_array = reinterpret_cast<T*>(sum_array_data);

  partial_sums += output_index * num_partial_buckets_per_output +
                  bucket_group_index * num_buckets_per_group * num_tiles;
  scalars += output_index * n * element_num_bytes;

  // initialize the bucket partial sums
  /* std::printf("num_buckets_per_group=%d\n", num_buckets_per_group); */
  /* std::printf("bucket_group_index=%d\n", bucket_group_index); */
  /* T* __restrict__ sums = sum_array + num_buckets_per_group * bucket_group_index; */
  T* sums = sum_array + num_buckets_per_group * bucket_group_index;
  sums[0] = T::identity();
  sums[1] = T::identity();
  sums[2] = T::identity();
  // TODO: why does above work but below doesn't?
#if 0
  for (unsigned sum_index = 0; sum_index < num_buckets_per_group; ++sum_index) {
    sums[sum_index] = T::identity();
  }
#endif

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
      add(sums[digit - 1u], sums[digit - 1u], e);
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
void compute_bucket_sums(basct::span<T> sums, const basdv::stream& stream,
                         basct::cspan<T> generators, basct::cspan<const uint8_t*> scalars,
                         unsigned element_num_bytes, unsigned bit_width) noexcept {
  auto num_outputs = scalars.size();
  auto num_buckets_per_group = (1u << bit_width) - 1u;
  auto num_bucket_groups = basn::divide_up(element_num_bytes * 8u, bit_width);
  auto num_buckets_per_output = num_buckets_per_group * num_bucket_groups;
  std::print("num_buckets_per_output={} num_bucket_groups={}\n", num_buckets_per_output,
             num_bucket_groups);
  auto num_buckets = num_buckets_per_output * num_outputs;
  auto n = static_cast<unsigned>(generators.size());
  if (n == 0) {
    return;
  }
  SXT_DEBUG_ASSERT(
      // clang-format off
      sums.size() == num_buckets
      // clang-format on
  );

  auto num_tiles = std::min(64u, n); // TODO: set better
  std::print("num_tiles = {}\n", num_tiles);                                     
  std::print("num_bucket_groups = {}\n", num_bucket_groups);
  std::print("element_num_bytes = {}\n", element_num_bytes);

  memr::async_device_resource resource{stream};

  // scalar_array
  memmg::managed_array<uint8_t> scalar_array{num_outputs * element_num_bytes * n, &resource};
  mtxb::make_device_scalar_array(scalar_array, stream, scalars, element_num_bytes, n);

  // set up generators
  memmg::managed_array<T> generators_dev{n, &resource};
  basdv::async_copy_host_to_device(generators_dev, generators, stream);

  // launch kernel
  std::print("num_buckets_per_output={}\n", num_buckets_per_output);
  auto shared_memory_num_bytes = num_buckets_per_output * sizeof(T);
  std::print("shared_memory_num_bytes = {}\n", shared_memory_num_bytes);
  memmg::managed_array<T> partial_sums{num_buckets * num_tiles, &resource};
  bucket_sum_kernel<<<dim3(num_outputs, num_tiles, 1), num_bucket_groups, shared_memory_num_bytes,
                      stream>>>(partial_sums.data(), generators_dev.data(), scalar_array.data(),
                                element_num_bytes, bit_width, n);
  generators_dev.reset();
  scalar_array.reset();

  // combine bucket sums
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
}
} // namespace sxt::mtxbk
