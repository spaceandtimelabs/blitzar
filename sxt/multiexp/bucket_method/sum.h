#pragma once

#include <cstddef>

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/device_viewable.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/multiexp/bucket_method/bucket_descriptor.h"
#include "sxt/multiexp/bucket_method/multiproduct_table.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// bucket_sum_kernel 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
CUDA_CALLABLE void bucket_sum_kernel(T* __restrict__ sums, const T* __restrict__ generators,
                                     const bucket_descriptor* __restrict__ table,
                                     const unsigned* __restrict__ indexes,
                                     unsigned sum_index) noexcept {
  auto descriptor = table[sum_index];
  if (descriptor.num_entries == 0) {
    sums[descriptor.bucket_index] = T::identity();
    return;
  }
  auto first = descriptor.entry_first;
  auto sum = generators[indexes[first]];
  for (unsigned i = 1; i < descriptor.num_entries; ++i) {
    auto e = generators[indexes[first + i]];
    add_inplace(sum, e);
  }
  sums[descriptor.bucket_index] = sum;
}

//--------------------------------------------------------------------------------------------------
// compute_bucket_sums
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> compute_bucket_sums(basct::span<T> sums, basct::cspan<T> generators,
                                   basct::cspan<const uint8_t*> scalars, unsigned element_num_bytes,
                                   unsigned bit_width) noexcept {
  SXT_DEBUG_ASSERT(
      basdv::is_host_pointer(sums.data())
  );
  auto num_buckets = sums.size();
  auto n = generators.size();
  if (n == 0) {
    co_return;
  }

  // set up the multiproduct table
  memmg::managed_array<bucket_descriptor> table{memr::get_device_resource()};
  memmg::managed_array<unsigned> indexes{memr::get_device_resource()};
  auto fut = compute_multiproduct_table(table, indexes, scalars, element_num_bytes, n, bit_width);

  memmg::managed_array<T> generators_data{memr::get_device_resource()};
  auto generators_dev = co_await xendv::make_active_device_viewable(generators_data, generators);
  co_await std::move(fut);
  SXT_DEBUG_ASSERT(table.size() == num_buckets);

  // compute bucket sums
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<T> sums_dev{num_buckets, &resource};
  auto f = [
    // clang-format off
    sums = sums_dev.data(),
    generators = generators_dev.data(),
    table = table.data(),
    indexes = indexes.data()
    // clang-format on
  ] __device__ __host__ (unsigned /*num_buckets*/, unsigned sum_index) noexcept {
    bucket_sum_kernel(sums, generators, table, indexes, sum_index);
  };
  algi::launch_for_each_kernel(stream, f, num_buckets);

  // copy sums to host
  basdv::async_copy_device_to_host(sums, sums_dev, stream);
  co_await xendv::await_stream(std::move(stream));
}
} // namespace sxt::mtxbk
