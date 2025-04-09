/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2025-present Space and Time Labs, Inc.
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

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/pinned_buffer.h"
#include "sxt/base/device/pinned_buffer2.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/synchronization.h"

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// generate_to_device_one_sweep
//--------------------------------------------------------------------------------------------------
template <class T, class F>
  requires requires(basct::span<T> buf, F f, size_t i) {
    { f(buf, i) } noexcept;
  }
xena::future<> generate_to_device_one_sweep(basct::span<T> dst, const basdv::stream& stream,
                                            F f) noexcept {
  if (dst.empty()) {
    co_return;
  }
  auto n = dst.size();
  auto num_bytes = n * sizeof(T);
  SXT_RELEASE_ASSERT(
      // clang-format off
      basdv::is_active_device_pointer(dst.data()) &&
      num_bytes <= basdv::pinned_buffer2::capacity()
      // clang-format on
  );
  basdv::pinned_buffer2 buffer(num_bytes);
  auto data = static_cast<T*>(buffer.data());
  f(basct::span<T>{data, n}, 0u);
  basdv::async_memcpy_host_to_device(static_cast<void*>(dst.data()), buffer.data(), num_bytes,
                                     stream);
  co_await await_stream(stream);
}

//--------------------------------------------------------------------------------------------------
// generate_to_device
//--------------------------------------------------------------------------------------------------
template <class T, class F>
  requires requires(basct::span<T> buffer, F f, size_t i) {
    { f(buffer, i) } noexcept;
  }
xena::future<> generate_to_device(basct::span<T> dst, const basdv::stream& stream, F f) noexcept {
  if (dst.empty()) {
    co_return;
  }
  auto n = dst.size();
  SXT_RELEASE_ASSERT(
      // clang-format off
      basdv::is_active_device_pointer(dst.data()) &&
      sizeof(T) < basdv::pinned_buffer::size()
      // clang-format on
  );
  auto num_bytes = n * sizeof(T);
  if (num_bytes <= basdv::pinned_buffer2::capacity()) {
    co_return co_await generate_to_device_one_sweep(dst, stream, f);
  }
  std::byte* out = reinterpret_cast<std::byte*>(dst.data());
  size_t pos = 0;

  auto fill_buffer = [&](basdv::pinned_buffer2& buffer) noexcept {
    auto data = static_cast<T*>(buffer.data());
    auto count = std::min(buffer.size() / sizeof(T), n - pos);
    f(basct::span<T>{data, count}, pos);
    pos += count;
    return count * sizeof(T);
  };

  // copy
  basdv::pinned_buffer2 cur_buffer(basdv::pinned_buffer2::capacity()),
      alt_buffer(basdv::pinned_buffer2::capacity());
  auto chunk_size = fill_buffer(cur_buffer);
  SXT_DEBUG_ASSERT(pos < n, "copy can't be done in a single sweep");
  while (pos < n) {
    basdv::async_memcpy_host_to_device(static_cast<void*>(out), cur_buffer.data(), chunk_size,
                                       stream);
    out += chunk_size;
    chunk_size = fill_buffer(alt_buffer);
    co_await await_stream(stream);
    std::swap(cur_buffer, alt_buffer);
  }
  basdv::async_memcpy_host_to_device(static_cast<void*>(out), cur_buffer.data(), chunk_size,
                                     stream);
  co_await await_stream(stream);
}
} // namespace sxt::xendv
