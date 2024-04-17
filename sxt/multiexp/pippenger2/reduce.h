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

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/type/raw_stream.h"

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
} // namespace sxt::mtxpp2
