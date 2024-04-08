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
#include <cstdint>

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/multiexp/pippenger2/partition_table_accessor.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// compute_partition_index
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE inline uint16_t compute_partition_index(const uint8_t* __restrict__ scalars,
                                                      unsigned step, unsigned n,
                                                      unsigned bit_index) noexcept {
  uint16_t res = 0;
  unsigned num_elements = std::min(16u, n);
  for (unsigned i = 0; i < num_elements; ++i) {
    auto byte = scalars[i * step];
    auto bit_value = byte & (1u << bit_index);
    res |= static_cast<uint16_t>(bit_value << i);
  }
  return res;
}

//--------------------------------------------------------------------------------------------------
// partition_product_kernel
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
__global__ void partition_product_kernel(T* __restrict__ products,
                                         const T* __restrict__ partition_table,
                                         const uint8_t* __restrict__ scalars, unsigned n) noexcept {
  constexpr unsigned num_partition_entries = 1u << 16u;
  auto byte_index = threadIdx.x;
  auto bit_offset = threadIdx.y;
  auto output_index = blockIdx.x;
  auto num_bytes_per_output = blockDim.x;
  auto num_outputs = gridDim.x;
  auto step = num_bytes_per_output * num_outputs;

  scalars += byte_index + output_index * num_bytes_per_output;
  products += 8u * num_bytes_per_output * output_index;
  products += byte_index * 8u + bit_offset;

  // lookup the first entry
  auto partition_index = compute_partition_index(scalars, step, n, bit_offset);
  auto res = partition_table[partition_index];

  // sum remaining entries
  while (n >= 16u) {
    n -= 16u;
    partition_table += num_partition_entries;
    scalars += 16u * step;

    partition_index = compute_partition_index(scalars, step, n, bit_offset);
    auto e = partition_table[partition_index];
    add_inplace(res, e);
  }

  // write result
  *products = res;
}

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
  SXT_DEBUG_ASSERT(offset % 16u == 0);

  // scalars_dev
  memmg::managed_array<uint8_t> scalars_dev{scalars.size(), memr::get_device_resource()};
  auto scalars_fut = [&]() noexcept -> xena::future<> {
    basdv::stream stream;
    basdv::async_copy_host_to_device(scalars_dev, scalars, stream);
    co_await xendv::await_stream(stream);
  };

  // partition_table
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<T> partition_table{num_partitions * num_table_entries, &resource};
  accessor.async_copy_precomputed_sums_to_device(partition_table, stream, offset / 16u);
  co_await std::move(scalars_fut);

  // product
}
} // namespace sxt::mtxpp2
