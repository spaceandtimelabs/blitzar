/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
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
#include "sxt/execution/device/computation_handle.h"

#include "sxt/base/device/stream.h"
#include "sxt/base/device/stream_handle.h"
#include "sxt/base/device/stream_pool.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/error/assert.h"

namespace sxt::xendv {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
computation_handle::computation_handle(computation_handle&& other) noexcept {
  head_ = other.head_;
  other.head_ = nullptr;
}

//--------------------------------------------------------------------------------------------------
// computation_handle
//--------------------------------------------------------------------------------------------------
computation_handle::~computation_handle() noexcept { this->wait(); }

//--------------------------------------------------------------------------------------------------
// operator=
//--------------------------------------------------------------------------------------------------
computation_handle& computation_handle::operator=(computation_handle&& other) noexcept {
  this->wait();
  head_ = other.head_;
  other.head_ = nullptr;
  return *this;
}

//--------------------------------------------------------------------------------------------------
// wait
//--------------------------------------------------------------------------------------------------
void computation_handle::wait() noexcept {
  if (head_ == nullptr) {
    return;
  }
  auto pool = basdv::get_stream_pool();
  do {
    auto handle = head_;
    SXT_DEBUG_ASSERT(handle->stream != nullptr);
    basdv::synchronize_stream(handle->stream);
    head_ = handle->next;
    handle->next = nullptr;
    pool->release_handle(handle);
  } while (head_ != nullptr);
}

//--------------------------------------------------------------------------------------------------
// add_stream
//--------------------------------------------------------------------------------------------------
void computation_handle::add_stream(basdv::stream&& stream) noexcept {
  auto handle = stream.release_handle();
  SXT_DEBUG_ASSERT(handle != nullptr && handle->next == nullptr);
  handle->next = head_;
  head_ = handle;
}
} // namespace sxt::xendv
