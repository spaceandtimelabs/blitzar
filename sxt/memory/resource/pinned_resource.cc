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
#include "sxt/memory/resource/pinned_resource.h"

#include <cuda_runtime.h>

#include <string>

#include "sxt/base/error/panic.h"

namespace sxt::memr {
//--------------------------------------------------------------------------------------------------
// do_allocate
//--------------------------------------------------------------------------------------------------
void* pinned_resource::do_allocate(size_t bytes, size_t /*alignment*/) noexcept {
  void* res;
  auto rcode = cudaMallocHost(&res, bytes);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMallocHost failed: {}", cudaGetErrorString(rcode));
  }
  return res;
}

//--------------------------------------------------------------------------------------------------
// do_deallocate
//--------------------------------------------------------------------------------------------------
void pinned_resource::do_deallocate(void* ptr, size_t /*bytes*/, size_t /*alignment*/) noexcept {
  auto rcode = cudaFreeHost(ptr);
  if (rcode != cudaSuccess) {
    baser::panic("cudaFreeHost failed: {}", cudaGetErrorString(rcode));
  }
}

//--------------------------------------------------------------------------------------------------
// do_is_equal
//--------------------------------------------------------------------------------------------------
bool pinned_resource::do_is_equal(const std::pmr::memory_resource& other) const noexcept {
  return this == &other;
}

//--------------------------------------------------------------------------------------------------
// get_pinned_resource
//--------------------------------------------------------------------------------------------------
std::pmr::memory_resource* get_pinned_resource() noexcept {
  // Use a heap allocated object that never gets deleted since we want this
  // to be available for the program duration.
  //
  // See https://google.github.io/styleguide/cppguide.html#Static_and_Global_Variables for
  // details on this use case.
  static pinned_resource* resource_p = new pinned_resource{};
  static std::pmr::unsynchronized_pool_resource* resource =
      new std::pmr::unsynchronized_pool_resource{resource_p};

  return resource;
}
} // namespace sxt::memr
