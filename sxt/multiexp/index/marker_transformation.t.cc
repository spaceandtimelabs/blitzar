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
#include "sxt/multiexp/index/marker_transformation.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/index/partition_marker_utility.h"

using namespace sxt;
using namespace sxt::mtxi;

TEST_CASE("we can transform an index table into partition markers") {
  auto consumer = [](basct::span<uint64_t>& indexes) noexcept {
    return consume_partition_marker(indexes, 2);
  };

  SECTION("we properly handle the empty table") {
    index_table tbl;
    REQUIRE(apply_marker_transformation(tbl.header(), consumer) == 0);
    REQUIRE(tbl.empty());
  }

  SECTION("we correctly handle a table with a single entry") {
    index_table tbl{{10}};
    REQUIRE(apply_marker_transformation(tbl.header(), consumer) == 1);
    index_table expected_tbl{{21}};
    REQUIRE(tbl == expected_tbl);
  }

  SECTION("we handle the case where a row shrinks in size") {
    index_table tbl{{2, 3, 4}};
    REQUIRE(apply_marker_transformation(tbl.header(), consumer) == 2);
    index_table expected_tbl{{7, 9}};
    REQUIRE(tbl == expected_tbl);
  }

  SECTION("we correctly handle multiple rows") {
    index_table tbl{{2, 3, 4}, {10}};
    REQUIRE(apply_marker_transformation(tbl.header(), consumer) == 3);
    index_table expected_tbl{{7, 9}, {21}};
    REQUIRE(tbl == expected_tbl);
  }

  SECTION("we can use an offset functor to skip over entries") {
    index_table tbl{{2, 3, 4}, {10}};
    auto offset_functor = [](basct::cspan<uint64_t> row) noexcept { return 1; };
    REQUIRE(apply_marker_transformation(tbl.header(), consumer, offset_functor) == 2);
    index_table expected_tbl{{2, 6, 9}, {10}};
    REQUIRE(tbl == expected_tbl);
  }
}
