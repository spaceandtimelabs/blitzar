#pragma once

#include <iostream>

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/iterator/index_range.h"
#include "sxt/base/iterator/index_range_utility.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/multiexp/pippenger2/partition_index.h"
#include "sxt/multiexp/pippenger2/product_kernel.h"
#include "sxt/multiexp/pippenger2/reduce.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// multiexponentiate_chunk 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> multiexponentiate_chunk(basct::span<T> res, basct::cspan<T> partition_table,
                                       basct::cspan<const uint8_t*> exponents,
                                       unsigned element_num_bytes, unsigned n) noexcept {
  auto num_outputs = exponents.size();
  auto num_partitions_per_product = basn::divide_up(n, 16u);
  auto num_products = num_outputs * element_num_bytes * 8u;

  memmg::managed_array<uint16_t> indexes{num_products * num_partitions_per_product,
                                         memr::get_device_resource()};

  // make the index table
  auto fut = fill_partition_indexes(indexes, exponents, element_num_bytes, n);

  // compute sums from indexes
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<T> partition_table_dev{partition_table.size(), &resource};
  basdv::async_copy_host_to_device(partition_table_dev, partition_table, stream);
  co_await std::move(fut);
  launch_product_kernel<0, T>(res.data(), stream, indexes.data(), partition_table_dev.data(),
                              num_products, num_partitions_per_product);
  co_await xendv::await_stream(stream);
}

//--------------------------------------------------------------------------------------------------
// multiexponentiate 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
xena::future<> multiexponentiate(basct::span<T> res, basct::cspan<T> partition_table,
                                 basct::cspan<const uint8_t*> exponents, unsigned element_num_bytes,
                                 unsigned n, unsigned max_partition_chunk_size = 16u) noexcept {
  auto num_outputs = res.size();
  auto num_products = num_outputs * element_num_bytes * 8u;
  auto num_partitions_per_product = basn::divide_up(n, 16u);

  // compute product chunks
  auto chunks = basit::split(
      basit::index_range{0, num_partitions_per_product}.max_chunk_size(max_partition_chunk_size),
      1u);
  std::cerr << "chunks: " << (chunks.second - chunks.first) << std::endl;
  (void)chunks;
  memmg::managed_array<T> products{num_products, memr::get_device_resource()};
  co_await multiexponentiate_chunk<T>(products, partition_table, exponents, element_num_bytes, n);

  // combine results
  basdv::stream stream;
  memr::async_device_resource resource{stream};

  // reduce
  memmg::managed_array<T> res_dev{num_outputs, &resource};
  reduce_products<T>(res_dev, stream, products);
  products.reset();
  basdv::async_copy_device_to_host(res, res_dev, stream);
  co_await xendv::await_stream(stream);
}
} // namespace sxt::mtxpp2
