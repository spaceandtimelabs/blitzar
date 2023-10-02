#pragma once

#include <algorithm>
#include <cstddef>

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/property.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/iterator/index_range.h"
#include "sxt/base/iterator/index_range_iterator.h"
#include "sxt/base/iterator/index_range_utility.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/device_viewable.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/managed_device_resource.h"
#include "sxt/multiexp/bucket_method/accumulate_kernel.h"
#include "sxt/multiexp/bucket_method/combination_kernel.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// make_exponents_viewable
//--------------------------------------------------------------------------------------------------
basct::cspan<const uint8_t>
make_exponents_viewable(memmg::managed_array<uint8_t>& exponents_viewable_data,
                        basct::cspan<const uint8_t*> exponents, const basit::index_range& rng,
                        const basdv::stream& stream) noexcept;

//--------------------------------------------------------------------------------------------------
// accumulate_buckets_impl
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> accumulate_buckets_impl(basct::span<T> bucket_sums, basct::cspan<T> generators,
                                       basct::cspan<const uint8_t*> exponents,
                                       basit::index_range rng) noexcept {
  unsigned n = rng.size();
  auto num_outputs = exponents.size();
  auto num_blocks = std::min(192u, n);
  static constexpr int num_bytes = 32; // hard code to 32 for now

  basdv::stream stream;

  // partial_bucket_sums
  memr::async_device_resource resource{stream};
  memmg::managed_array<T> partial_bucket_sums{bucket_sums.size() * num_blocks, &resource};

  // generators_viewable
  memmg::managed_array<T> generators_viewable_data{&resource};
  auto generators_viewable = xendv::make_active_device_viewable(
      generators_viewable_data, generators.subspan(rng.a(), rng.size()));

  // exponents_viewable
  memmg::managed_array<uint8_t> exponents_viewable_data{&resource};
  auto exponents_viewable =
      make_exponents_viewable(exponents_viewable_data, exponents, rng, stream);

  // partial bucket accumulation kernel
  xendv::synchronize_event(stream, generators_viewable);
  bucket_accumulate<<<dim3(num_blocks, num_outputs, 1), num_bytes, 0, stream>>>(
      partial_bucket_sums.data(), generators_viewable.value().data(), exponents_viewable.data(), n);
  generators_viewable_data.reset();
  exponents_viewable_data.reset();


  // combine partial sums
  memmg::managed_array<T> bucket_sums_dev{bucket_sums.size(), &resource};
  combine_partial_bucket_sums<<<dim3(255, num_outputs, 1), num_bytes, 0, stream>>>(
      bucket_sums_dev.data(), partial_bucket_sums.data(), num_blocks);
  partial_bucket_sums.reset();

  // add buckets
  memmg::managed_array<T> bucket_sums_host{bucket_sums.size(), memr::get_managed_device_resource()};
  basdv::async_copy_device_to_host(bucket_sums_host, bucket_sums_dev, stream);
  co_await xendv::await_stream(stream);
  for (size_t bucket_index=0; bucket_index<bucket_sums.size(); ++bucket_index) {
    add_inplace(bucket_sums[bucket_index], bucket_sums_host[bucket_index]);
  }
}

template <bascrv::element T>
xena::future<> accumulate_buckets_impl2(basct::span<T> bucket_sums, basct::cspan<T> generators,
                                        basct::cspan<const uint8_t*> exponents,
                                        basit::index_range rng) noexcept {
  unsigned n = rng.size();
  auto num_outputs = exponents.size();
  auto num_blocks = std::min(192u, n);
  static constexpr int num_bytes = 32; // hard code to 32 for now

  basdv::stream stream;

  // partial_bucket_sums
  memr::async_device_resource resource{stream};
  memmg::managed_array<T> partial_bucket_sums{bucket_sums.size() * num_blocks, &resource};

  // generators_viewable
  memmg::managed_array<T> generators_viewable_data{&resource};
  auto generators_viewable = xendv::make_active_device_viewable(
      generators_viewable_data, generators.subspan(rng.a(), rng.size()));

  // exponents_viewable
  memmg::managed_array<uint8_t> exponents_viewable_data{&resource};
  auto exponents_viewable =
      make_exponents_viewable(exponents_viewable_data, exponents, rng, stream);

  // partial bucket accumulation kernel
  xendv::synchronize_event(stream, generators_viewable);
  bucket_accumulate<<<dim3(num_blocks, num_outputs, 1), num_bytes, 0, stream>>>(
      partial_bucket_sums.data(), generators_viewable.value().data(), exponents_viewable.data(), n);
  generators_viewable_data.reset();
  exponents_viewable_data.reset();


  // combine partial sums
  memmg::managed_array<T> bucket_sums_dev{bucket_sums.size(), &resource};
  combine_partial_bucket_sums<<<dim3(255, num_outputs, 1), num_bytes, 0, stream>>>(
      bucket_sums_dev.data(), partial_bucket_sums.data(), num_blocks);
  partial_bucket_sums.reset();
  basdv::async_copy_to_device(bucket_sums, bucket_sums_dev, stream);
  co_await xendv::await_stream(stream);
}

//--------------------------------------------------------------------------------------------------
// accumulate_buckets
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> accumulate_buckets(basct::span<T> bucket_sums, basct::cspan<T> generators,
                                  basct::cspan<const uint8_t*> exponents) noexcept {
  constexpr size_t bucket_group_size = 255;
  constexpr size_t num_bucket_groups = 32;
  auto num_outputs = exponents.size();
  SXT_DEBUG_ASSERT(
      bucket_sums.size() == bucket_group_size * num_bucket_groups * num_outputs
  );
  for (auto& e : bucket_sums) {
    e = T::identity();
  }
  co_await xendv::concurrent_for_each(
      basit::index_range{0, generators.size()}, [&](const basit::index_range& rng) noexcept {
        return accumulate_buckets_impl(bucket_sums, generators, exponents, rng);
      });
}

template <bascrv::element T>
xena::future<> accumulate_buckets2(basct::span<T> bucket_sums, basct::cspan<T> generators,
                                  basct::cspan<const uint8_t*> exponents) noexcept {
  constexpr size_t bucket_group_size = 255;
  constexpr size_t num_bucket_groups = 32;
  static constexpr int num_bytes = 32; // hard code to 32 for now
  auto num_outputs = exponents.size();
  SXT_DEBUG_ASSERT(
      bucket_sums.size() == bucket_group_size * num_bucket_groups * num_outputs &&
      basdv::is_active_device_pointer(bucket_sums.data())
  );
  auto [first, last] =
      basit::split(basit::index_range{0, generators.size()}, basdv::get_num_devices());
  auto num_chunks = std::distance(first, last);

  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<T> partial_bucket_sums_data{&resource};
  basct::span<T> partial_bucket_sums;
  if (num_chunks > 1) {
    partial_bucket_sums_data.resize(num_bucket_groups * bucket_group_size * num_chunks);
    partial_bucket_sums = partial_bucket_sums_data;
  } else {
    partial_bucket_sums = bucket_sums;
  }

  size_t step = bucket_group_size * num_bucket_groups;
  size_t i = 0;
  co_await xendv::concurrent_for_each(first, last, [&](const basit::index_range& rng) noexcept {
    return accumulate_buckets_impl2(partial_bucket_sums.subspan(i++ * step, step), generators,
                                    exponents, rng);
  });
  if (num_chunks <= 1) {
    co_return;
  }
  combine_partial_bucket_sums<<<dim3(255, num_outputs, 1), num_bytes, 0, stream>>>(
      bucket_sums.data(), partial_bucket_sums.data(), num_chunks);
  co_await xendv::await_stream(stream);
}
} // namespace sxt::mtxbk
