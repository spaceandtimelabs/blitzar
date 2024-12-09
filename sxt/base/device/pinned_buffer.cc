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
#include "sxt/base/device/pinned_buffer.h"

#include "sxt/base/device/pinned_buffer_pool.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// consructor
//--------------------------------------------------------------------------------------------------
pinned_buffer::pinned_buffer() noexcept : handle_{get_pinned_buffer_pool()->aquire_handle()} {}

pinned_buffer::pinned_buffer(pinned_buffer&& ptr) noexcept : handle_{ptr.handle_} {
  ptr.handle_ = nullptr;
}

//--------------------------------------------------------------------------------------------------
// destructor
//--------------------------------------------------------------------------------------------------
pinned_buffer::~pinned_buffer() noexcept {
  if (handle_ != nullptr) {
    get_pinned_buffer_pool()->release_handle(handle_);
  }
}

//--------------------------------------------------------------------------------------------------
// operator=
//--------------------------------------------------------------------------------------------------
pinned_buffer& pinned_buffer::operator=(pinned_buffer&& ptr) noexcept {
  if (handle_ != nullptr) {
    get_pinned_buffer_pool()->release_handle(handle_);
  }
  handle_ = ptr.handle_;
  ptr.handle_ = nullptr;
  return *this;
}

//--------------------------------------------------------------------------------------------------
// size
//--------------------------------------------------------------------------------------------------
size_t pinned_buffer::size() noexcept { return pinned_buffer_size; }
} // namespace sxt::basdv
