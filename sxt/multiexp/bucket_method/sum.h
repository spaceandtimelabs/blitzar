#pragma once

#include <vector>

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/container/span.h"
#include "sxt/base/container/span_utility.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/property.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/iterator/index_range.h"
#include "sxt/base/iterator/index_range_iterator.h"
#include "sxt/base/iterator/index_range_utility.h"
#include "sxt/base/log/log.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/bucket_method/multiproduct_table.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// sum_bucket 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
CUDA_CALLABLE void sum_bucket(T* __restrict__ sums, const T* __restrict__ generators,
                              const uint16_t* __restrict__ bucket_prefix_counts,
                              const uint16_t* __restrict__ indexes, unsigned num_buckets_per_digit,
                              unsigned n, unsigned index) noexcept {
  auto digit_index = index / num_buckets_per_digit;
  auto bucket_offset = index % num_buckets_per_digit;

  // adjust pointers
  auto& sum = sums[index];
  bucket_prefix_counts += digit_index * num_buckets_per_digit;
  indexes += digit_index * n;

  // sum
  uint16_t first;
  if (bucket_offset == 0) {
    first = 0;
  } else {
    first = bucket_prefix_counts[bucket_offset - 1u];
  }
  auto last = bucket_prefix_counts[bucket_offset];
  if (first == last) {
    sum = T::identity();
    return;
  }
  T e = generators[indexes[first++]];
  for (; first != last; ++first) {
    auto t = generators[indexes[first]];
    add_inplace(e, t);
  }
  sum = e;
}

//--------------------------------------------------------------------------------------------------
// sum_buckets 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> sum_buckets(basct::span<T> sums, basct::cspan<T> generators,
                           basct::cspan<const uint8_t*> exponents, unsigned element_num_bytes,
                           unsigned bit_width) noexcept {
  auto num_buckets_per_digit = (1u << bit_width) - 1u;
  auto num_digits = basn::divide_up(element_num_bytes * 8u, bit_width);  
  auto num_outputs = static_cast<unsigned>(exponents.size());
  auto num_buckets_total = static_cast<unsigned>(sums.size());
  auto n = static_cast<unsigned>(generators.size());
  SXT_DEBUG_ASSERT(basdv::is_active_device_pointer(sums.data()));

  // compute multiproduct table
  memmg::managed_array<uint16_t> bucket_prefix_counts{num_buckets_total,
                                                      memr::get_device_resource()};
  memmg::managed_array<uint16_t> indexes{n * num_digits * num_outputs, memr::get_device_resource()};
  auto fut = make_multiproduct_table(bucket_prefix_counts, indexes, exponents, element_num_bytes,
                                     bit_width, n);

  // copy generators to device
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<T> generators_dev{n, &resource};
  basdv::async_copy_host_to_device(generators_dev, generators, stream);

  // sum buckets
  memmg::managed_array<T> sums_dev{num_buckets_total, &resource};
  co_await std::move(fut);
  basl::info("summing {} buckets", num_buckets_total);
  auto f = [
               // clang-format off
    sums = sums.data(),
    generators = generators_dev.data(),
    bucket_prefix_counts = bucket_prefix_counts.data(),
    indexes = indexes.data(),
    num_buckets_per_digit = num_buckets_per_digit,
    n = n
               // clang-format on
  ] __device__
           __host__(unsigned /*num_buckets_total*/, unsigned index) noexcept {
             sum_bucket<T>(sums, generators, bucket_prefix_counts, indexes, num_buckets_per_digit,
                           n, index);
           };
  algi::launch_for_each_kernel(stream, f, num_buckets_total);
  co_await xendv::await_stream(stream);
}
} // namespace sxt::mtxbk
