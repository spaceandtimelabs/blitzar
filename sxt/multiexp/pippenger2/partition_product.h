#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/multiexp/pippenger2/partition_table_accessor.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// partition_product 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> partition_product(basct::span<T> products,
                                 const partition_table_accessor<T>& accessor,
                                 basct::cspan<uint8_t> scalars, unsigned offset) noexcept {
  auto num_products = products.size();
  auto n = scalars.size() * 8u / num_products;
  auto num_partitions = basn::divide_up(n, 16u);
  auto num_table_entries = 1u << 16u;
  SXT_DEBUG_ASSERT(
      offset % 16u == 0
  );

  // scalars_dev
  memmg::managed_array<uint8_t> scalars_dev{scalars.size(), memr::get_device_resource()};
  auto scalars_fut = [&]() noexcept -> xena::future<> {
    basdv::stream stream;
    basdv::async_copy_host_to_device(scalars_dev, scalars, stream);
    co_await xendv::await_stream(stream);
  };

  // partition_table
  basdv::stream stream;
  memmg::managed_array<T> partition_table{num_partitions * num_table_entries,
                                          memr::get_device_resource()};
  accessor.async_copy_precomputed_sums_to_device(partition_table, stream, offset / 16u);
  co_await std::move(scalars_fut);

  // product
}
} // namespace sxt::mtxpp2
