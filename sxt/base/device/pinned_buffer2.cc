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
#include "sxt/base/device/pinned_buffer2.h"

#include "sxt/base/device/pinned_buffer_pool.h"
#include "sxt/base/error/assert.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
pinned_buffer2::pinned_buffer2(size_t size) noexcept
    : handle_{get_pinned_buffer_pool()->acquire_handle()}, size_{size} {
  SXT_RELEASE_ASSERT(size_ <= this->capacity());
}

pinned_buffer2::pinned_buffer2(pinned_buffer2&& other) noexcept
    : handle_{std::exchange(other.handle_, nullptr)}, size_{std::exchange(other.size_, 0)} {}

//--------------------------------------------------------------------------------------------------
// destructor
//--------------------------------------------------------------------------------------------------
pinned_buffer2::~pinned_buffer2() noexcept {
  if (handle_ != nullptr) {
    get_pinned_buffer_pool()->release_handle(handle_);
  }
}

//--------------------------------------------------------------------------------------------------
// operator=
//--------------------------------------------------------------------------------------------------
pinned_buffer2& pinned_buffer2::operator=(pinned_buffer2&& other) noexcept {
  this->reset();
  handle_ = std::exchange(other.handle_, nullptr);
  size_ = std::exchange(other.size_, 0);
  return *this;
}

//--------------------------------------------------------------------------------------------------
// capacity
//--------------------------------------------------------------------------------------------------
size_t pinned_buffer2::capacity() noexcept { return pinned_buffer_size; }

//--------------------------------------------------------------------------------------------------
// resize
//--------------------------------------------------------------------------------------------------
void pinned_buffer2::resize(size_t size) noexcept {
  SXT_RELEASE_ASSERT(size <= this->capacity());
  if (handle_ == nullptr) {
    handle_ = get_pinned_buffer_pool()->acquire_handle();
  }
  size_ = size;
}

//--------------------------------------------------------------------------------------------------
// fill
//--------------------------------------------------------------------------------------------------
basct::cspan<std::byte> pinned_buffer2::fill_from_host(basct::cspan<std::byte> src) noexcept {
  if (src.empty()) {
    return src;
  }
  if (handle_ == nullptr) {
    handle_ = get_pinned_buffer_pool()->acquire_handle();
  }
  auto n = std::min(src.size(), this->capacity() - size_);
  std::copy_n(src.data(), n, static_cast<std::byte*>(handle_->ptr) + size_);
  size_ += n;
  return src.subspan(n);
}

//--------------------------------------------------------------------------------------------------
// reset
//--------------------------------------------------------------------------------------------------
void pinned_buffer2::reset() noexcept {
  if (handle_ == nullptr) {
    return;
  }
  get_pinned_buffer_pool()->release_handle(handle_);
  handle_ = nullptr;
  size_ = 0;
}
} // namespace sxt::basdv
