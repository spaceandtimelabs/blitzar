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
#include "sxt/multiexp/pippenger2/partition_index_kernel.h"

#include <vector>

#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can compute the partition indexes for a multiexponentiation") {
  std::pmr::vector<uint16_t> indexes{memr::get_managed_device_resource()};
  std::pmr::vector<uint8_t> scalars{memr::get_managed_device_resource()};

  indexes.resize(256);
  scalars.resize(32);

  std::pmr::vector<uint16_t> expected(indexes.size());

  basdv::stream stream;

  SECTION("we handle a single scalar of zero") {
    launch_fill_partition_indexes_kernel(indexes.data(), stream, scalars.data(), 1, 1);
    basdv::synchronize_stream(stream);
    REQUIRE(indexes == expected);
  }

  SECTION("we handle a single scalar of one") {
    scalars[0] = 1;
    launch_fill_partition_indexes_kernel(indexes.data(), stream, scalars.data(), 1, 1);
    basdv::synchronize_stream(stream);
    expected[0] = 1;
    REQUIRE(indexes == expected);
  }

  SECTION("we handle two scalars") {
    scalars.resize(32 * 2);
    scalars[0] = 1;
    scalars[32] = 1;
    launch_fill_partition_indexes_kernel(indexes.data(), stream, scalars.data(), 1, 2);
    basdv::synchronize_stream(stream);
    expected[0] = 3;
    REQUIRE(indexes == expected);
  }
}
