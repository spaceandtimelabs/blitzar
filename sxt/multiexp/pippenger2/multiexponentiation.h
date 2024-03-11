#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/multiexp/pippenger2/partition_index.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// multiexponentiate 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> multiexponentiate(basct::span<T> res, basct::cspan<T> partition_table,
                                 basct::cspan<const uint8_t*> exponents,
                                 unsigned element_num_bytes,
                                 unsigned n) noexcept {
  auto num_outputs = res.size();
  auto num_partitions_per_product = basn::divide_up(n, 16u);

  memmg::managed_array<uint16_t> indexes{num_outputs * element_num_bytes * 8u *
                                             num_partitions_per_product,
                                         memr::get_device_resource()};

  // make the index table
  auto fut = fill_partition_indexes(indexes, exponents, element_num_bytes, n);

  // compute sums from indexes
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<T> product_table_dev{n * (1u << 16u), &resource};

  co_await std::move(fut);

  // reduce
  (void)res;
  (void)partition_table;
  (void)exponents;
  (void)element_num_bytes;
}
} // namespace sxt::mtxpp2
