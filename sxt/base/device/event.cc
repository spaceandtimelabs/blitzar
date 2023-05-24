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
#include "sxt/base/device/event.h"

#include <cuda_runtime.h>

#include <string>
#include <utility>

#include "sxt/base/error/panic.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
event::event() noexcept {
  auto rcode = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
  if (rcode != cudaSuccess) {
    baser::panic("cudaEventCreate failed: " + std::string(cudaGetErrorString(rcode)));
  }
}

event::event(event&& other) noexcept : event_{std::exchange(other.event_, nullptr)} {}

//--------------------------------------------------------------------------------------------------
// assignment
//--------------------------------------------------------------------------------------------------
event& event::operator=(event&& other) noexcept {
  this->clear();
  event_ = std::exchange(other.event_, nullptr);
  return *this;
}

//--------------------------------------------------------------------------------------------------
// clear
//--------------------------------------------------------------------------------------------------
void event::clear() noexcept {
  if (event_ == nullptr) {
    return;
  }
  auto rcode = cudaEventDestroy(event_);
  if (rcode != cudaSuccess) {
    baser::panic("cudaEventDestroy failed: " + std::string(cudaGetErrorString(rcode)));
  }
  event_ = nullptr;
}

//--------------------------------------------------------------------------------------------------
// query_is_ready
//--------------------------------------------------------------------------------------------------
bool event::query_is_ready() noexcept {
  auto rcode = cudaEventQuery(event_);
  if (rcode == cudaSuccess) {
    return true;
  }
  if (rcode == cudaErrorNotReady) {
    return false;
  }
  baser::panic("cudaEventQuery failed: " + std::string(cudaGetErrorString(rcode)));
}

//--------------------------------------------------------------------------------------------------
// destructor
//--------------------------------------------------------------------------------------------------
event::~event() noexcept { this->clear(); }
} // namespace sxt::basdv
