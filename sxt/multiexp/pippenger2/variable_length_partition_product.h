#pragma once

#include <concepts>

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/base/num/round_up.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/multiexp/pippenger2/partition_table_accessor.h"
#include "sxt/multiexp/pippenger2/partition_product.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// async_partition_product
//--------------------------------------------------------------------------------------------------
/**
 * Compute the multiproduct for the bits of an array of scalars of varying lengths using an accessor 
 * to precomputed sums for each group of generators.
 */
template <bascrv::element T, class U>
  requires std::constructible_from<T, U>
xena::future<> async_partition_product(basct::span<T> products_slice, unsigned num_products,
                                       const partition_table_accessor<U>& accessor,
                                       basct::cspan<uint8_t> scalars,
                                       basct::cspan<unsigned> lengths, unsigned offset) noexcept {
  auto num_products_round_8 = basn::round_up<size_t>(num_products, 8u);
  auto n = static_cast<unsigned>(scalars.size() * 8u / num_products_round_8);
  auto window_width = accessor.window_width();
  auto num_partitions = basn::divide_up(n, window_width);
  auto partition_table_size = 1u << window_width;
  SXT_DEBUG_ASSERT(
      // clang-format off
      products_slice.size() <= num_products &&
      products_slice.size() == lengths.size() &&
      offset % window_width == 0 &&
      scalars.size() * 8u % num_products_round_8 == 0 &&
      basdv::is_active_device_pointer(products_slice.data()) &&
      basdv::is_host_pointer(scalars.data())
      // clang-format on
  );

  // scalars_dev
  memmg::managed_array<uint8_t> scalars_dev{scalars.size(), memr::get_device_resource()};
  auto scalars_fut = [&]() noexcept -> xena::future<> {
    basdv::stream stream;
    basdv::async_copy_host_to_device(scalars_dev, scalars, stream);
    co_await xendv::await_stream(stream);
  }();

  // lengths_dev
  memmg::managed_array<unsigned> lengths_dev{lengths.size(), memr::get_device_resource()};
  auto lengths_fut = [&]() noexcept -> xena::future<> {
    basdv::stream stream;
    basdv::async_copy_host_to_device(lengths_dev, lengths, stream);
    co_await xendv::await_stream(stream);
  }();

  // partition_table
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<U> partition_table{num_partitions * partition_table_size, &resource};
  accessor.async_copy_to_device(partition_table, stream, offset / window_width);
  co_await std::move(lengths_fut);
  co_await std::move(scalars_fut);

  // product
  auto f = [
               // clang-format off
    products = products_slice.data(),
    scalars = scalars_dev.data(),
    partition_table = partition_table.data(),
    window_width = window_width,
    num_products = num_products,
    lengths = lengths_dev.data()
               // clang-format on
  ] __device__
           __host__(unsigned num_slice_products, unsigned product_index) noexcept {
             auto n = lengths[product_index];
             auto& product = products[product_index];
             product_index += num_products - num_slice_products;
             auto byte_index = product_index / 8u;
             auto bit_offset = product_index % 8u;
             auto num_products_round_8 = basn::round_up(num_products, 8u);
             partition_product_kernel<T>(product, partition_table, scalars, byte_index, bit_offset,
                                         window_width, num_products_round_8, n);
           };
  algi::launch_for_each_kernel(stream, f, products_slice.size());
  co_await xendv::await_stream(stream);
}

//--------------------------------------------------------------------------------------------------
// partition_product
//--------------------------------------------------------------------------------------------------
/**
 * Compute the multiproduct for the bits of an array of scalars using an accessor to
 * precomputed sums for each group of generators.
 */
template <bascrv::element T, class U>
  requires std::constructible_from<T, U>
void partition_product(basct::span<T> products_slice, unsigned num_products,
                       const partition_table_accessor<U>& accessor, basct::cspan<uint8_t> scalars,
                       basct::cspan<unsigned> lengths, unsigned offset) noexcept {
  auto num_slice_products = products_slice.size();
  auto num_products_round_8 = basn::round_up<size_t>(num_products, 8u);
  auto n = static_cast<unsigned>(scalars.size() * 8u / num_products_round_8);
  auto window_width = accessor.window_width();
  auto num_partitions = basn::divide_up(n, window_width);
  auto partition_table_size = 1u << window_width;
  SXT_DEBUG_ASSERT(
      // clang-format off
      num_slice_products <= num_products &&
      num_slice_products == lengths.size() &&
      offset % window_width == 0 &&
      scalars.size() * 8u % num_products_round_8 == 0
      // clang-format on
  );
  std::pmr::monotonic_buffer_resource alloc;

  auto partition_table = accessor.host_view(&alloc, offset, num_partitions * partition_table_size);

  for (unsigned product_index = 0; product_index < num_slice_products; ++product_index) {
    auto len = lengths[product_index];
    auto& product = products_slice[product_index];
    product_index += num_products - num_slice_products;
    auto byte_index = product_index / 8u;
    auto bit_offset = product_index % 8u;
    auto num_products_round_8 = basn::round_up(num_products, 8u);
    partition_product_kernel<T>(product, partition_table.data(), scalars.data(), byte_index,
                                bit_offset, window_width, num_products_round_8, len);
  }
}
} // namespace sxt::mtxpp2
