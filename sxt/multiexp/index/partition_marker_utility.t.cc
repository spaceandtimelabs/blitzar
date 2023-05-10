/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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
#include "sxt/multiexp/index/partition_marker_utility.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::mtxi;

TEST_CASE("we can condense a sequences of indexes down to a marker of its partition "
          "set") {
  SECTION("we correctly handle cases with a single element") {
    uint64_t data[1];

    basct::span<uint64_t> values;
    uint64_t marker;

    *data = 0;
    values = data;
    marker = consume_partition_marker(values, 1);
    REQUIRE(values.empty());
    REQUIRE(marker == 1);

    *data = 1;
    values = data;
    marker = consume_partition_marker(values, 1);
    REQUIRE(values.empty());
    REQUIRE(marker == 3);

    *data = 5;
    values = data;
    marker = consume_partition_marker(values, 10);
    REQUIRE(values.empty());
    REQUIRE(marker == 32);
  }

  SECTION("we correctly handle cases with two elements with two elements") {
    uint64_t data[2];

    basct::span<uint64_t> values;
    uint64_t marker;

    data[0] = 0;
    data[1] = 1;
    values = data;
    marker = consume_partition_marker(values, 1);
    REQUIRE(values.size() == 1);
    REQUIRE(values[0] == 1);
    REQUIRE(marker == 1);

    data[0] = 0;
    data[1] = 1;
    values = data;
    marker = consume_partition_marker(values, 2);
    REQUIRE(values.empty());
    REQUIRE(marker == 3);

    data[0] = 1;
    data[1] = 2;
    values = data;
    marker = consume_partition_marker(values, 2);
    REQUIRE(values.size() == 1);
    REQUIRE(values[0] == 2);
    REQUIRE(marker == 2);

    data[0] = 3;
    data[1] = 5;
    values = data;
    marker = consume_partition_marker(values, 2);
    REQUIRE(values.size() == 1);
    REQUIRE(values[0] == 5);
    REQUIRE(marker == 6);
  }
}
