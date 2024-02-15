#pragma once

#include <vector>

#include "sxt/base/container/span.h"
#include "sxt/base/container/span_utility.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/property.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/iterator/index_range.h"
#include "sxt/base/iterator/index_range_iterator.h"
#include "sxt/base/iterator/index_range_utility.h"
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
                              const uint16_t* __restrict__ indexes, unsigned index) noexcept {
  (void)sums;
  (void)bucket_prefix_counts;
  (void)indexes;
}

//--------------------------------------------------------------------------------------------------
// sum_buckets_chunk 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> sum_buckets_chunk(basct::span<T> sums, basct::cspan<T> generators,
                                 basct::cspan<const uint8_t*> exponents, unsigned element_num_bytes,
                                 unsigned bit_width) noexcept {
  auto num_digits = basn::divide_up(element_num_bytes * 8u, bit_width);  
  auto num_outputs = static_cast<unsigned>(exponents.size());
  auto num_buckets_total = static_cast<unsigned>(sums.size());
  auto n = static_cast<unsigned>(generators.size());

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

  // compute bucket sums
  memmg::managed_array<T> sums_dev{num_buckets_total, &resource};
  co_await std::move(fut);
  basdv::async_copy_device_to_host(sums, sums_dev, stream);
  co_await xendv::await_stream(stream);
}

//--------------------------------------------------------------------------------------------------
// sum_buckets 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> sum_buckets(basct::span<T> sums, basct::cspan<T> generators,
                           basct::cspan<const uint8_t*> exponents, unsigned element_num_bytes,
                           unsigned bit_width) noexcept {
  auto num_outputs = exponents.size();
  auto num_buckets_total = sums.size();
  auto n = generators.size();
  auto rng = basit::index_range{0, n}.min_chunk_size(1024).max_chunk_size(1024);
  auto split_factor = basdv::get_num_devices() * 2u;
  auto [chunk_first, chunk_last] = basit::split(rng, split_factor);
  auto num_chunks = static_cast<unsigned>(chunk_last - chunk_first);
  memmg::managed_array<T> partial_sums{sums.size() * num_chunks, memr::get_pinned_resource()};
  unsigned chunk_index = 0;
  std::vector<const uint8_t*> chunk_exponents(num_outputs);

  // compute partial sums
  co_await xendv::concurrent_for_each(
      chunk_first, chunk_last, [&](const basit::index_range& rng) noexcept -> xena::future<> {
        auto chunk_generators = generators.subspan(rng.a(), rng.size());
        for (unsigned output_index=0; output_index<num_outputs; ++output_index) {
          chunk_exponents[output_index] = exponents[output_index] + rng.a() * element_num_bytes +
                                          output_index * element_num_bytes * n;
        }
        auto sums_slice =
            basct::subspan(partial_sums, rng.a() * num_buckets_total, num_buckets_total);
        co_await sum_buckets_chunk<T>(sums_slice, chunk_generators, chunk_exponents,
                                      element_num_bytes, bit_width);
      });

  // combine partial sums
}
} // namespace sxt::mtxbk
