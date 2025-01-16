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
#include "sxt/execution/device/copy.h"

#include <cassert>
#include <cstring>

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/generate.h"

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// strided_copy_host_to_device
//--------------------------------------------------------------------------------------------------
xena::future<> strided_copy_host_to_device(std::byte* dst, const basdv::stream& stream,
                                           const std::byte* src, size_t n, size_t count,
                                           size_t stride) noexcept {
  SXT_RELEASE_ASSERT(
      // clang-format off
      basdv::is_active_device_pointer(dst) &&
      basdv::is_host_pointer(src) &&
      stride >= n
      // clang-format on
  );
  auto num_bytes = n * count;
  auto f = [&](basct::span<std::byte> buffer, size_t index) noexcept {
    auto remaining_size = buffer.size();
    auto out = buffer.data();
    auto slice_index = index / n;
    auto slice_pos = index - n * slice_index;

    auto data = src + n * slice_index + slice_pos;

    auto chunk_size = std::min(remaining_size, n - slice_pos);
    out = std::copy_n(data, chunk_size, out);
    data += stride;
    remaining_size -= chunk_size;

    while (remaining_size > 0) {
      chunk_size = std::min(remaining_size, n);
      out = std::copy_n(data, chunk_size, out);
      data += stride;
      remaining_size -= chunk_size;
    }
  };
  co_await generate_to_device(basct::span<std::byte>{dst, num_bytes}, stream, f);
}
} // namespace sxt::xendv
