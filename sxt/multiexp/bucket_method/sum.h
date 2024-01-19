#pragma once

#include <cstddef>

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/device_viewable.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/multiexp/bucket_method/bucket_descriptor.h"
#include "sxt/multiexp/bucket_method/multiproduct_table.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// compute_bucket_sums 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> compute_bucket_sums(basct::span<T> sums, basct::cspan<T> generators,
                                   basct::cspan<const uint8_t*> scalars, unsigned element_num_bytes,
                                   unsigned bit_width) noexcept {
  auto num_buckets = sums.size();
  auto n = generators.size();
  memmg::managed_array<bucket_descriptor> table{memr::get_device_resource()};
  memmg::managed_array<unsigned> indexes{memr::get_device_resource()};
  auto fut = compute_multiproduct_table(table, indexes, scalars, element_num_bytes, n, bit_width);

  memmg::managed_array<T> generators_data{memr::get_device_resource()};
  auto generators_dev = co_await xendv::make_active_device_viewable(generators_data, generators);
  co_await std::move(fut);

  SXT_DEBUG_ASSERT(table.size() == num_buckets);

  auto f = [
               // clang-format off
    sums_p = sums.data(), 
    generators_p = generators_dev.data(), 
    table = table.data(),
    indexes = indexes.data()
               // clang-format on
  ] __device__(unsigned /*num_buckets*/, unsigned bucket_index) noexcept {
    T* __restrict__ sums = sums_p;
    const T* __restrict__ generators = generators_p;
    auto descriptor = table[bucket_index];
    if (descriptor.num_entries == 0) {
      return;
    }
    auto first = descriptor.entry_first;
    auto sum = generators[indexes[first]];
    for (unsigned i = 1; i < descriptor.num_entries; ++i) {
      auto e = generators[indexes[first + i]];
      add_inplace(sum, e);
    }
    sums[descriptor.bucket_index] = sum;
  };
  co_await algi::for_each(f, num_buckets);
}
} // namespace sxt::mtxbk
