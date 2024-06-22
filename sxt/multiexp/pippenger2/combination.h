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
#include "sxt/base/container/span_utility.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/type/raw_stream.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// combine_impl
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
CUDA_CALLABLE void combine_impl(T* __restrict__ reduction, const T* __restrict__ elements,
                                unsigned step, unsigned reduction_size) noexcept {
  T res = elements[0];
  for (unsigned i = 1; i < reduction_size; ++i) {
    auto e = elements[step * i];
    add_inplace(res, e);
  }
  *reduction = res;
}

//--------------------------------------------------------------------------------------------------
// combine
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void combine(basct::span<T> res, bast::raw_stream_t stream, basct::cspan<T> elements) noexcept {
  auto n = static_cast<unsigned>(res.size());
  SXT_DEBUG_ASSERT(
      // clang-format off
      elements.size() >= n && 
      elements.size() % n == 0 &&
      basdv::is_active_device_pointer(res.data()) &&
      basdv::is_active_device_pointer(elements.data())
      // clang-format on
  );
  auto reduction_size = static_cast<unsigned>(elements.size() / n);
  auto f = [
               // clang-format off
    reductions = res.data(),
    elements = elements.data(),
    reduction_size = reduction_size
               // clang-format on
  ] __device__
           __host__(unsigned n, unsigned index) noexcept {
             combine_impl(reductions + index, elements + index, n, reduction_size);
           };
  algi::launch_for_each_kernel(stream, f, n);
}
} // namespace sxt::mtxpp2
