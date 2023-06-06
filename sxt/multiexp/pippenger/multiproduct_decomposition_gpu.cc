/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
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
#include "sxt/multiexp/pippenger/multiproduct_decomposition_gpu.h"

#include <algorithm>
#include <memory>

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/pippenger/multiproduct_decomposition_kernel.h"

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// compute_multiproduct_decomposition
//--------------------------------------------------------------------------------------------------
xena::future<> compute_multiproduct_decomposition(memmg::managed_array<unsigned>& indexes,
                                                  basct::span<unsigned> product_sizes,
                                                  const basdv::stream& stream,
                                                  mtxb::exponent_sequence exponents) noexcept {
  auto element_num_bytes = exponents.element_nbytes;
  auto element_num_bits = 8u * element_num_bytes;
  SXT_DEBUG_ASSERT(
      // clang-format off
      product_sizes.size() == element_num_bits &&
      basdv::is_host_pointer(product_sizes.data())
      // clang-format on
  );
  auto n = exponents.n;
  std::fill(product_sizes.begin(), product_sizes.end(), 0);
  indexes.reset();
  if (n == 0) {
    co_return;
  }

  memr::async_device_resource resource{stream};

  // set up exponents
  memmg::managed_array<uint8_t> exponents_data{&resource};
  if (!basdv::is_device_pointer(exponents.data)) {
    exponents_data = memmg::managed_array<uint8_t>{
        n * element_num_bytes,
        &resource,
    };
    basdv::async_memcpy_host_to_device(exponents_data.data(), exponents.data, n * element_num_bytes,
                                       stream);
    exponents.data = exponents_data.data();
  }

  memmg::managed_array<unsigned> block_counts;
  co_await count_exponent_bits(block_counts, stream, exponents);

  // compute product sizes
  SXT_DEBUG_ASSERT(block_counts.size() % element_num_bits == 0);
  unsigned num_blocks = block_counts.size() / element_num_bits;
  unsigned num_one_bits = 0;
  for (unsigned bit_index = 0; bit_index < element_num_bits; ++bit_index) {
    for (unsigned block_index = 0; block_index < num_blocks; ++block_index) {
      auto& cnt = block_counts[block_index * element_num_bits + bit_index];
      product_sizes[bit_index] += cnt;
      auto t = num_one_bits;
      num_one_bits += cnt;
      cnt = t;
    }
  }
  if (num_one_bits == 0) {
    co_return;
  }

  // rearrange indexes
  indexes = memmg::managed_array<unsigned>{
      num_one_bits,
      indexes.get_allocator(),
  };
  SXT_DEBUG_ASSERT(basdv::is_device_pointer(indexes.data()));
  co_await decompose_exponent_bits(indexes, stream, block_counts, exponents);
}
} // namespace sxt::mtxpi
