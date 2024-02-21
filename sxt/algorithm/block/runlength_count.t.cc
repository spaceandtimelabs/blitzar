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
#include "sxt/algorithm/block/runlength_count.h"

#include <cstdint>
#include <vector>

#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::algbk;

template <size_t NumThreads> static __global__ void k1(int* counts) {
  uint8_t items[1];
  items[0] = 1;
  using RunlengthCount = runlength_count<uint8_t, int, NumThreads, 256>;
  __shared__ typename RunlengthCount::temp_storage temp_storage;
  auto cnts = RunlengthCount{temp_storage}.count(items);
  __syncthreads();
  for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) {
    counts[i] = cnts[i];
  }
}

template <size_t NumThreads> static __global__ void k2(int* counts) {
  uint8_t items[1];
  items[0] = threadIdx.x;
  using RunlengthCount = runlength_count<uint8_t, int, NumThreads, 256>;
  __shared__ typename RunlengthCount::temp_storage temp_storage;
  auto cnts = RunlengthCount{temp_storage}.count(items);
  __syncthreads();
  for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) {
    counts[i] = cnts[i];
  }
}

TEST_CASE("we can count the run lengths of sorted values") {
  std::pmr::vector<int> counts{256, memr::get_managed_device_resource()};

  std::pmr::vector<int> expected(counts.size());

  SECTION("we handle a single thread") {
    k1<1><<<1, 1>>>(counts.data());
    basdv::synchronize_device();
    expected[1] = 1;
    REQUIRE(counts == expected);
  }

  SECTION("we handle two threads") {
    k1<2><<<1, 2>>>(counts.data());
    basdv::synchronize_device();
    expected[1] = 2;
    REQUIRE(counts == expected);
  }

  SECTION("we handle two threads with different values") {
    k2<2><<<1, 2>>>(counts.data());
    basdv::synchronize_device();
    expected[0] = 1;
    expected[1] = 1;
    REQUIRE(counts == expected);
  }
}
