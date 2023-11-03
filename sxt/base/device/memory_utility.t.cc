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
#include "sxt/base/device/memory_utility.h"

#include <cuda_runtime.h>

#include <memory>
#include <vector>

#include "sxt/base/container/span.h"
#include "sxt/base/device/pointer_attributes.h"
#include "sxt/base/device/state.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::basdv;

TEST_CASE("we can determine if a pointer is device or host") {
  stream stream;

  SECTION("we handle host-side pointers") {
    auto ptr = std::make_unique<int>(123);
    REQUIRE(!is_active_device_pointer(ptr.get()));
  }

  SECTION("we handle device pointers") {
    void* ptr;
    REQUIRE(cudaMalloc(&ptr, 100) == cudaSuccess);
    REQUIRE(is_active_device_pointer(ptr));
    pointer_attributes attrs;
    get_pointer_attributes(attrs, ptr);
    REQUIRE(attrs.kind == pointer_kind_t::device);
    REQUIRE(attrs.device == get_device());
    cudaFree(ptr);
  }

  SECTION("we handle managed pointers") {
    void* ptr;
    REQUIRE(cudaMallocManaged(&ptr, 100) == cudaSuccess);
    REQUIRE(is_active_device_pointer(ptr));
    pointer_attributes attrs;
    get_pointer_attributes(attrs, ptr);
    REQUIRE(attrs.kind == pointer_kind_t::managed);
    cudaFree(ptr);
  }

  SECTION("we can copy memory") {
    std::vector<int> v = {1, 2, 3};
    std::vector<int> w(v.size());
    int* data;
    REQUIRE(cudaMalloc(&data, sizeof(int) * v.size()) == cudaSuccess);
    basct::span<int> buffer{data, v.size()};
    async_copy_host_to_device(buffer, v, stream);
    async_copy_device_to_host(w, buffer, stream);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    REQUIRE(w == v);
    REQUIRE(cudaFree(data) == cudaSuccess);
  }

  SECTION("we can copy a host or device point to device memory") {
    std::vector<int> v = {1, 2, 3};
    std::vector<int> w(v.size());
    int* data;
    REQUIRE(cudaMalloc(&data, sizeof(int) * v.size()) == cudaSuccess);
    basct::span<int> buffer{data, v.size()};
    async_copy_to_device(buffer, v, stream);
    async_copy_device_to_host(w, buffer, stream);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    REQUIRE(w == v);
    REQUIRE(cudaFree(data) == cudaSuccess);
  }
}
