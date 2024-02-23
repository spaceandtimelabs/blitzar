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
#include "sxt/multiexp/base/scalar_array.h"

#include <numeric>
#include <vector>

#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::mtxb;

TEST_CASE("we can copy transpose scalar arrays to device memory") {
  memmg::managed_array<uint8_t> array{memr::get_managed_device_resource()};

  SECTION("we can transpose a single scalar of 1 byte") {
    std::vector<uint8_t> scalars1(1);
    scalars1[0] = 123u;
    array.resize(1);
    std::vector<const uint8_t*> scalars = {scalars1.data()};
    auto fut = transpose_scalars_to_device(array, scalars, 1,  1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(array[0] == 123u);
  }

  SECTION("we can transpose a single scalar of 2 bytes") {
    std::vector<uint8_t> scalars1(2);
    scalars1[0] = 1u;
    scalars1[1] = 2u;
    array.resize(2);
    std::vector<const uint8_t*> scalars = {scalars1.data()};
    auto fut = transpose_scalars_to_device(array, scalars, 2,  1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(array[0] == 1u);
    REQUIRE(array[1] == 2u);
  }

  SECTION("we can transpose a single scalar of 32 bytes") {
    std::vector<uint8_t> scalars1(32);
    std::iota(scalars1.begin(), scalars1.end(), 0);
    array.resize(32);
    std::vector<const uint8_t*> scalars = {scalars1.data()};
    auto fut = transpose_scalars_to_device(array, scalars, 32,  1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(array[0] == 0u);
    REQUIRE(array[31u] == 31u);
  }

  SECTION("we can transpose two scalars of 1 byte") {
    std::vector<uint8_t> scalars1(2);
    std::iota(scalars1.begin(), scalars1.end(), 0);
    array.resize(2);
    std::vector<const uint8_t*> scalars = {scalars1.data()};
    auto fut = transpose_scalars_to_device(array, scalars, 1, 2);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(array[0] == 0u);
    REQUIRE(array[1] == 1u);
  }

  SECTION("we can transpose two scalars of 2 byte") {
    std::vector<uint8_t> scalars1(4);
    std::iota(scalars1.begin(), scalars1.end(), 0);
    array.resize(4);
    std::vector<const uint8_t*> scalars = {scalars1.data()};
    auto fut = transpose_scalars_to_device(array, scalars, 2, 2);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(array[0] == 0u);
    REQUIRE(array[1] == 2u);
    REQUIRE(array[2] == 1u);
    REQUIRE(array[3] == 3u);
  }

  SECTION("we can transpose two scalars of 32 bytes") {
    std::vector<uint8_t> scalars1(64);
    std::iota(scalars1.begin(), scalars1.end(), 0);
    array.resize(64);
    std::vector<const uint8_t*> scalars = {scalars1.data()};
    auto fut = transpose_scalars_to_device(array, scalars, 32, 2);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(array[0] == 0u);
    REQUIRE(array[1] == 32u);
    REQUIRE(array[2] == 1u);
    REQUIRE(array[3] == 33u);
    REQUIRE(array[62] == 31u);
    REQUIRE(array[63] == 63u);
  }
}
