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
#include "sxt/base/device/stream_pool.h"

#include <cuda_runtime.h>

#include <string>

#include "sxt/base/device/active_device_guard.h"
#include "sxt/base/device/stream_handle.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/error/panic.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// make_stream_handle
//--------------------------------------------------------------------------------------------------
static stream_handle* make_stream_handle(int device) noexcept {
  active_device_guard active_guard{device};
  auto res = new stream_handle{};
  auto rcode = cudaStreamCreate(&res->stream);
  if (rcode != cudaSuccess) {
    baser::panic("cudaStreamCreate failed: " + std::string(cudaGetErrorString(rcode)));
  }
  res->next = nullptr;
  return res;
};

//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
stream_pool::stream_pool(size_t initial_size) noexcept {
  for (int device = 0; device < heads_.size(); ++device) {
    auto& head = heads_[device];
    for (size_t i = 0; i < initial_size; ++i) {
      auto handle = make_stream_handle(device);
      handle->next = head;
      head = handle;
    }
  }
}

//--------------------------------------------------------------------------------------------------
// destructor
//--------------------------------------------------------------------------------------------------
stream_pool::~stream_pool() noexcept {
  for (auto handle : heads_) {
    while (handle != nullptr) {
      auto next = handle->next;
      auto rcode = cudaStreamDestroy(handle->stream);
      if (rcode != cudaSuccess) {
        baser::panic("cudaStreamDestroy failed: " + std::string(cudaGetErrorString(rcode)));
      }
      delete handle;
      handle = next;
    }
  }
}

//--------------------------------------------------------------------------------------------------
// aquire_handle
//--------------------------------------------------------------------------------------------------
stream_handle* stream_pool::aquire_handle(int device) noexcept {
  auto& head = heads_[device];
  if (head == nullptr) {
    return make_stream_handle(device);
  }
  auto res = head;
  head = res->next;
  res->next = nullptr;
  return res;
}

//--------------------------------------------------------------------------------------------------
// release_handle
//--------------------------------------------------------------------------------------------------
void stream_pool::release_handle(stream_handle* handle) noexcept {
  auto& head = heads_[handle->device];
  SXT_DEBUG_ASSERT(handle != nullptr && handle->next == nullptr);
  handle->next = head;
  head = handle;
}

//--------------------------------------------------------------------------------------------------
// get_stream_pool
//--------------------------------------------------------------------------------------------------
stream_pool* get_stream_pool(size_t initial_size) noexcept {
  // Allocate a thread local pool that's available for the duration of the process.
  static thread_local auto pool = new stream_pool{initial_size};
  return pool;
}
} // namespace sxt::basdv
