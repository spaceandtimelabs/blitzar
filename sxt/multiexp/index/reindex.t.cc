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
#include "sxt/multiexp/index/reindex.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/multiexp/index/index_table.h"

using namespace sxt;
using namespace sxt::mtxi;

TEST_CASE("we can reindex a table") {
  std::vector<uint64_t> values_data(10);
  basct::span<uint64_t> values{values_data.data(), values_data.size()};

  SECTION("reindexing an empty table does nothing") {
    index_table tbl;

    reindex_rows(tbl.header(), values);
    REQUIRE(tbl.empty());
    REQUIRE(values.empty());
  }

  SECTION("we can reindex a table with a single element") {
    index_table tbl{{10}};

    reindex_rows(tbl.header(), values);
    values_data.resize(values.size());
    index_table expected_tbl{{0}};
    REQUIRE(tbl == expected_tbl);
    REQUIRE(values_data == std::vector<uint64_t>{10});
  }

  SECTION("we can reindex a table with two elements") {
    index_table tbl{{10, 15}};

    reindex_rows(tbl.header(), values);
    values_data.resize(values.size());
    index_table expected_tbl{{0, 1}};
    REQUIRE(tbl == expected_tbl);
    REQUIRE(values_data == std::vector<uint64_t>{10, 15});
  }

  SECTION("we correctly handle two rows with identical values") {
    index_table tbl{{10}, {10}};

    reindex_rows(tbl.header(), values);
    values_data.resize(values.size());
    index_table expected_tbl{{0}, {0}};
    REQUIRE(tbl == expected_tbl);
    REQUIRE(values_data == std::vector<uint64_t>{10});
  }

  SECTION("we can use an offset functor to skip over entries") {
    index_table tbl{{10, 11}, {10, 12}};
    auto f = [](basct::cspan<uint64_t> row) noexcept { return 1; };
    reindex_rows(tbl.header(), values, f);
    values_data.resize(values.size());
    index_table expected_tbl{{10, 0}, {10, 1}};
    REQUIRE(tbl == expected_tbl);
    REQUIRE(values_data == std::vector<uint64_t>{11, 12});
  }

  SECTION("we correctly handle two rows with two unique values") {
    index_table tbl{{10}, {20}};

    reindex_rows(tbl.header(), values);
    values_data.resize(values.size());
    index_table expected_tbl{{0}, {1}};
    REQUIRE(tbl == expected_tbl);
    REQUIRE(values_data == std::vector<uint64_t>{10, 20});
  }

  SECTION("we correctly handle two rows with three unique values") {
    index_table tbl{{10}, {1, 20}};

    reindex_rows(tbl.header(), values);
    values_data.resize(values.size());
    index_table expected_tbl{{1}, {0, 2}};
    REQUIRE(tbl == expected_tbl);
    REQUIRE(values_data == std::vector<uint64_t>{1, 10, 20});
  }

  SECTION("we correctly handle multiple rows with multiple unique values") {
    index_table tbl{{101}, {1, 4, 20}, {3, 7}, {4, 100, 101}};

    reindex_rows(tbl.header(), values);
    values_data.resize(values.size());
    index_table expected_tbl{{6}, {0, 2, 4}, {1, 3}, {2, 5, 6}};
    REQUIRE(tbl == expected_tbl);
    REQUIRE(values_data == std::vector<uint64_t>{1, 3, 4, 7, 20, 100, 101});
  }
}
