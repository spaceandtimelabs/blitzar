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
#include "sxt/multiexp/pippenger2/partition_index.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can compute the partition indexes that correspond to a multiexponentiation") {

  std::pmr::vector<uint16_t> indexes{memr::get_managed_device_resource()};
  indexes.resize(256);
  std::vector<const uint8_t*> scalars;

  std::pmr::vector<uint16_t> expected;
  expected.resize(256);

  SECTION("we handle the case of a single element of zero") {
    std::vector<uint8_t> scalars1(32);
    scalars = {scalars1.data()};
    auto fut = fill_partition_indexes(indexes, scalars, 32, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(indexes == expected);
  }

  SECTION("we handle the case of a single element of one") {
    std::vector<uint8_t> scalars1(32);
    scalars1[0] = 1;
    scalars = {scalars1.data()};
    auto fut = fill_partition_indexes(indexes, scalars, 32, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    expected[0] = 1;
    REQUIRE(indexes == expected);
  }
}
