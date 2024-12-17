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

#include "sxt/base/device/pinned_buffer_handle.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// pinned_buffer
//--------------------------------------------------------------------------------------------------
class pinned_buffer {
public:
  pinned_buffer() noexcept;
  pinned_buffer(pinned_buffer&& ptr) noexcept;
  pinned_buffer(const pinned_buffer&) noexcept = delete;

  ~pinned_buffer() noexcept;

  pinned_buffer& operator=(pinned_buffer&& ptr) noexcept;
  pinned_buffer& operator=(const pinned_buffer& ptr) noexcept = delete;

  static size_t size() noexcept;

  void* data() noexcept { return handle_->ptr; }

  const void* data() const noexcept { return handle_->ptr; }

  operator void*() noexcept { return handle_->ptr; }

  operator const void*() const noexcept { return handle_->ptr; }

private:
  pinned_buffer_handle* handle_;
};
} // namespace sxt::basdv
