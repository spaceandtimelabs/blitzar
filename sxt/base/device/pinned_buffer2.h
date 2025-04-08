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

#include <cstddef>

#include "sxt/base/container/span.h"
#include "sxt/base/device/pinned_buffer_handle.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// pinned_buffer2
//--------------------------------------------------------------------------------------------------
class pinned_buffer2 {
public:
  pinned_buffer2() noexcept = default;
  pinned_buffer2(const pinned_buffer2&) noexcept = delete;
  pinned_buffer2(pinned_buffer2&& other) noexcept;

  ~pinned_buffer2() noexcept;

  pinned_buffer2& operator=(const pinned_buffer2&) noexcept = delete;
  pinned_buffer2& operator=(pinned_buffer2&& other) noexcept;

  bool empty() const noexcept { return size_ == 0; }

  bool full() const noexcept { return size_ == this->capacity(); }

  size_t size() const noexcept { return size_; }

  static size_t capacity() noexcept;

  void* data() noexcept {
    if (handle_ == nullptr) {
      return nullptr;
    }
    return handle_->ptr;
  }

  const void* data() const noexcept {
    if (handle_ == nullptr) {
      return nullptr;
    }
    return handle_->ptr;
  }

  basct::cspan<std::byte> fill_from_host(basct::cspan<std::byte> src) noexcept;

  void reset() noexcept;

private:
  pinned_buffer_handle* handle_ = nullptr;
  size_t size_ = 0;
};
} // namespace sxt::basdv
