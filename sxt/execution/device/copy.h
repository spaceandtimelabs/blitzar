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

#include <cstddef>

#include "sxt/base/container/span.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/async/future.h"

namespace sxt::basdv {
class stream;
}

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// strided_copy_host_to_device
//--------------------------------------------------------------------------------------------------
xena::future<> strided_copy_host_to_device(std::byte* dst, const basdv::stream& stream,
                                           const std::byte* src, size_t n, size_t count,
                                           size_t stride) noexcept;

template <class T>
xena::future<> strided_copy_host_to_device(basct::span<T> dst, const basdv::stream& stream,
                                           basct::cspan<T> src, size_t stride, size_t slice_size,
                                           size_t offset) noexcept {
  if (slice_size == 0) {
    SXT_RELEASE_ASSERT(dst.empty());
    return xena::make_ready_future();
  }
  auto count = dst.size() / slice_size;
  SXT_RELEASE_ASSERT(
      // clang-format off
      stride >= slice_size &&
      dst.size() == count * slice_size && 
      src.size() >= offset + (count - 1u)*stride + slice_size
      // clang-format on
  );
  return strided_copy_host_to_device(reinterpret_cast<std::byte*>(dst.data()), stream,
                                     reinterpret_cast<const std::byte*>(src.data() + offset),
                                     slice_size * sizeof(T), count, stride * sizeof(T));
}
} // namespace sxt::xendv
