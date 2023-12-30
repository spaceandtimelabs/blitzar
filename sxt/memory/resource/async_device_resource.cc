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
#include "sxt/memory/resource/async_device_resource.h"

#include <cuda_runtime.h>

#include "sxt/base/error/panic.h"

namespace sxt::memr {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
async_device_resource::async_device_resource(bast::raw_stream_t stream) noexcept
    : stream_{stream} {}

//--------------------------------------------------------------------------------------------------
// do_allocate
//--------------------------------------------------------------------------------------------------
void* async_device_resource::do_allocate(size_t bytes, size_t alignment) noexcept {
  void* res;
  auto rcode = cudaMallocAsync(&res, bytes, stream_);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMallocAsync failed: {}", cudaGetErrorString(rcode));
  }
  return res;
}

//--------------------------------------------------------------------------------------------------
// do_deallocate
//--------------------------------------------------------------------------------------------------
void async_device_resource::do_deallocate(void* ptr, size_t bytes, size_t alignment) noexcept {
  auto rcode = cudaFreeAsync(ptr, stream_);
  if (rcode != cudaSuccess) {
    baser::panic("cudaFreeAsync failed: {}", cudaGetErrorString(rcode));
  }
}

//--------------------------------------------------------------------------------------------------
// do_is_equal
//--------------------------------------------------------------------------------------------------
bool async_device_resource::do_is_equal(const std::pmr::memory_resource& other) const noexcept {
  return this == &other;
}
} // namespace sxt::memr
