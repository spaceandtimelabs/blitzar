/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <memory_resource>

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/base/num/round_up.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/multiexp/pippenger2/constants.h"
#include "sxt/multiexp/pippenger2/partition_table_accessor.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// compute_partition_index
//--------------------------------------------------------------------------------------------------
/**
 * Compute the index in the partition table of the product for a group of 16 scalars.
 */
CUDA_CALLABLE inline uint16_t compute_partition_index(const uint8_t* __restrict__ scalars,
                                                      unsigned step, unsigned window_width,
                                                      unsigned n, unsigned bit_index) noexcept {
  (void)window_width;
  uint16_t res = 0;
  unsigned num_elements = std::min(16u, n);
  auto mask = 1u << bit_index;
  for (unsigned i = 0; i < num_elements; ++i) {
    auto byte = scalars[i * step];
    auto bit_value = static_cast<uint16_t>((byte & mask) != 0);
    res |= static_cast<uint16_t>(bit_value << i);
  }
  return res;
}

//--------------------------------------------------------------------------------------------------
// partition_product_kernel
//--------------------------------------------------------------------------------------------------
template <bascrv::element T, class U>
  requires std::constructible_from<T, U>
CUDA_CALLABLE void
partition_product_kernel(T* __restrict__ products, const U* __restrict__ partition_table,
                         const uint8_t* __restrict__ scalars, unsigned byte_index,
                         unsigned bit_offset, unsigned window_width, unsigned num_products,
                         unsigned n) noexcept {
  auto num_partition_entries = 1u << window_width;

  auto step = num_products / 8u;

  scalars += byte_index;
  products += byte_index * 8u + bit_offset;

  // lookup the first entry
  auto partition_index = compute_partition_index(scalars, step, window_width, n, bit_offset);
  T res{partition_table[partition_index]};

  // sum remaining entries
  while (n > window_width) {
    n -= window_width;
    partition_table += num_partition_entries;
    scalars += window_width * step;

    partition_index = compute_partition_index(scalars, step, window_width, n, bit_offset);
    T e{partition_table[partition_index]};
    add_inplace(res, e);
  }

  // write result
  *products = res;
}

//--------------------------------------------------------------------------------------------------
// async_partition_product
//--------------------------------------------------------------------------------------------------
/**
 * Compute the multiproduct for the bits of an array of scalars using an accessor to
 * precomputed sums for each group of generators.
 */
template <bascrv::element T, class U>
  requires std::constructible_from<T, U>
xena::future<> async_partition_product(basct::span<T> products,
                                       const partition_table_accessor<U>& accessor,
                                       basct::cspan<uint8_t> scalars, unsigned offset) noexcept {
  auto num_products = products.size();
  auto num_products_round_8 = basn::round_up<size_t>(num_products, 8u);
  auto n = static_cast<unsigned>(scalars.size() * 8u / num_products_round_8);
  auto window_width = accessor.window_width();
  auto num_partitions = basn::divide_up(n, window_width);
  auto partition_table_size = 1u << window_width;
  SXT_DEBUG_ASSERT(
      // clang-format off
      offset % window_width == 0 &&
      scalars.size() * 8u % num_products_round_8 == 0 &&
      basdv::is_active_device_pointer(products.data()) &&
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

  // partition_table
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<U> partition_table{num_partitions * partition_table_size, &resource};
  accessor.async_copy_to_device(partition_table, stream, offset / window_width);
  co_await std::move(scalars_fut);

  // product
  auto f = [
               // clang-format off
    products = products.data(),
    scalars = scalars_dev.data(),
    partition_table = partition_table.data(),
    window_width = window_width,
    n = n
               // clang-format on
  ] __device__
           __host__(unsigned num_products, unsigned product_index) noexcept {
             auto byte_index = product_index / 8u;
             auto bit_offset = product_index % 8u;
             auto num_products_round_8 = basn::round_up(num_products, 8u);
             partition_product_kernel<T>(products, partition_table, scalars, byte_index, bit_offset,
                                         window_width, num_products_round_8, n);
           };
  algi::launch_for_each_kernel(stream, f, num_products);
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
void partition_product(basct::span<T> products, const partition_table_accessor<U>& accessor,
                       basct::cspan<uint8_t> scalars, unsigned offset) noexcept {
  auto num_products = products.size();
  auto num_products_round_8 = basn::round_up<size_t>(num_products, 8u);
  auto n = static_cast<unsigned>(scalars.size() * 8u / num_products_round_8);
  auto window_width = accessor.window_width();
  auto partition_table_size = 1u << window_width;
  SXT_DEBUG_ASSERT(
      // clang-format off
      scalars.size() * 8u % num_products_round_8 == 0 &&
      offset % window_width == 0
      // clang-format on
  );
  std::pmr::monotonic_buffer_resource alloc;

  auto partition_table =
      accessor.host_view(&alloc, offset, basn::divide_up(n, window_width) * partition_table_size);

  for (unsigned product_index = 0; product_index < num_products; ++product_index) {
    auto byte_index = product_index / 8u;
    auto bit_offset = product_index % 8u;
    auto num_products_round_8 = basn::round_up<size_t>(num_products, 8u);
    partition_product_kernel<T>(products.data(), partition_table.data(), scalars.data(), byte_index,
                                bit_offset, window_width, num_products_round_8, n);
  }
}
} // namespace sxt::mtxpp2
