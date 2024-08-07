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

#include <numeric>

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/type/raw_stream.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// reduce_output
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
CUDA_CALLABLE void reduce_output(T* __restrict__ reduction, const T* __restrict__ products,
                                 unsigned n) noexcept {
  T res = products[n - 1];
  --n;
  while (n-- > 0) {
    double_element(res, res);
    auto e = products[n];
    add_inplace(res, e);
  }
  *reduction = res;
}

//--------------------------------------------------------------------------------------------------
// reduce_products
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void reduce_products(basct::span<T> reductions, bast::raw_stream_t stream,
                     basct::cspan<T> products) noexcept {
  auto num_outputs = reductions.size();
  auto reduction_size = products.size() / reductions.size();
  SXT_DEBUG_ASSERT(
      // clang-format off
      basdv::is_active_device_pointer(reductions.data()) &&
      products.size() == reduction_size * num_outputs &&
      basdv::is_active_device_pointer(products.data())
      // clang-format on
  );
  auto f = [
               // clang-format off
    reductions = reductions.data(),
    products = products.data(),
    reduction_size = reduction_size
               // clang-format on
  ] __device__
           __host__(unsigned /*num_outputs*/, unsigned output_index) noexcept {
             reduce_output(reductions + output_index, products + output_index * reduction_size,
                           reduction_size);
           };
  algi::launch_for_each_kernel(stream, f, num_outputs);
}

template <bascrv::element T>
void reduce_products(basct::span<T> reductions, bast::raw_stream_t stream,
                     basct::cspan<unsigned> output_bit_table, basct::cspan<T> products) noexcept {
  auto num_outputs = reductions.size();
  SXT_DEBUG_ASSERT(
      // clang-format off
      basdv::is_active_device_pointer(reductions.data()) &&
      reductions.size() == num_outputs &&
      output_bit_table.size() == num_outputs &&
      basdv::is_active_device_pointer(products.data())
      // clang-format on
  );

  memr::async_device_resource resource{stream};

  // make partial bit table sums
  memmg::managed_array<unsigned> bit_table_partial_sums{num_outputs, memr::get_pinned_resource()};
  std::partial_sum(output_bit_table.begin(), output_bit_table.end(),
                   bit_table_partial_sums.begin());
  SXT_DEBUG_ASSERT(products.size() == bit_table_partial_sums[num_outputs - 1]);
  memmg::managed_array<unsigned> bit_table_partial_sums_dev{num_outputs, &resource};
  basdv::async_copy_host_to_device(bit_table_partial_sums_dev, bit_table_partial_sums, stream);

  // reduce products
  auto f = [
               // clang-format off
    reductions = reductions.data(),
    bit_table_partial_sums = bit_table_partial_sums_dev.data(),
    products = products.data()
               // clang-format on
  ] __device__
           __host__(unsigned /*num_outputs*/, unsigned output_index) noexcept {
             auto lookup_index = max(0, static_cast<int>(output_index) - 1);
             auto offset =
                 bit_table_partial_sums[lookup_index] * static_cast<unsigned>(output_index != 0);
             auto reduction_size = bit_table_partial_sums[output_index] - offset;
             reduce_output(reductions + output_index, products + offset, reduction_size);
           };
  algi::launch_for_each_kernel(stream, f, num_outputs);
}

template <bascrv::element T>
void reduce_products(basct::span<T> reductions, basct::cspan<T> products) noexcept {
  auto num_outputs = reductions.size();
  auto reduction_size = products.size() / reductions.size();
  SXT_DEBUG_ASSERT(products.size() == reduction_size * num_outputs);
  for (unsigned output_index = 0; output_index < num_outputs; ++output_index) {
    reduce_output(reductions.data() + output_index, products.data() + output_index * reduction_size,
                  reduction_size);
  }
}

template <bascrv::element T>
void reduce_products(basct::span<T> reductions, basct::cspan<unsigned> output_bit_table,
                     basct::cspan<T> products) noexcept {
  auto num_outputs = reductions.size();
  SXT_DEBUG_ASSERT(
      // clang-format off
      reductions.size() == num_outputs &&
      output_bit_table.size() == num_outputs
      // clang-format on
  );
  unsigned offset = 0;
  for (unsigned output_index = 0; output_index < num_outputs; ++output_index) {
    auto reduction_size = output_bit_table[output_index];
    reduce_output(reductions.data() + output_index, products.data() + offset, reduction_size);
    offset += reduction_size;
  }
}
} // namespace sxt::mtxpp2
