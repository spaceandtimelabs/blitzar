/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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
#include "sxt/base/device/stream.h"

#include "sxt/base/device/stream_handle.h"
#include "sxt/base/device/stream_pool.h"
#include "sxt/base/error/assert.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
stream::stream() noexcept { handle_ = get_stream_pool()->aquire_handle(); }

stream::stream(stream&& other) noexcept { handle_ = other.release_handle(); }

//--------------------------------------------------------------------------------------------------
// destructor
//--------------------------------------------------------------------------------------------------
stream::~stream() noexcept {
  if (handle_ == nullptr) {
    return;
  }
  get_stream_pool()->release_handle(handle_);
}

//--------------------------------------------------------------------------------------------------
// operator=
//--------------------------------------------------------------------------------------------------
stream& stream::operator=(stream&& other) noexcept {
  if (handle_ != nullptr) {
    get_stream_pool()->release_handle(handle_);
  }
  handle_ = other.release_handle();
  return *this;
}

//--------------------------------------------------------------------------------------------------
// release_handle
//--------------------------------------------------------------------------------------------------
stream_handle* stream::release_handle() noexcept {
  auto res = handle_;
  handle_ = nullptr;
  return res;
}

//--------------------------------------------------------------------------------------------------
// raw_stream
//--------------------------------------------------------------------------------------------------
CUstream_st* stream::raw_stream() const noexcept {
  SXT_DEBUG_ASSERT(handle_ != nullptr);
  return handle_->stream;
}
} // namespace sxt::basdv
