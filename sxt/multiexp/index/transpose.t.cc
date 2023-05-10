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
#include "sxt/multiexp/index/transpose.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/multiexp/index/index_table.h"

using namespace sxt;
using namespace sxt::mtxi;

TEST_CASE("we can transpose an index table") {
  index_table table_p;

  SECTION("we can transpose an empty index table") {
    index_table table{};
    REQUIRE(transpose(table_p, table.cheader(), 0) == 0);
    REQUIRE(table_p == index_table{});
  }

  SECTION("we can transpose an index table with a single entry") {
    index_table table{{0}};
    REQUIRE(transpose(table_p, table.cheader(), 1) == 0);
    REQUIRE(table_p == index_table{{0}});
  }

  SECTION("we can transpose a table with a single row") {
    index_table table{{0, 1, 2, 3}};
    REQUIRE(transpose(table_p, table.cheader(), 4) == 3);
    REQUIRE(table_p == index_table{{0}, {0}, {0}, {0}});
  }

  SECTION("we can transpose a table with two rows") {
    index_table table{{0, 1}, {0, 2, 3}};
    REQUIRE(transpose(table_p, table.cheader(), 4) == 3);
    REQUIRE(table_p == index_table{{0, 1}, {0}, {1}, {1}});
  }

  SECTION("we can add padding while transposing") {
    index_table table{{0, 1}, {0, 2, 3}};
    REQUIRE(transpose(table_p, table.cheader(), 4, 2) == 3);
    REQUIRE(table_p == index_table{{0, 0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 1}});
  }

  SECTION("we can transpose a table with an empty row") {
    index_table table{{0, 1}, {}, {0, 2, 3}};
    REQUIRE(transpose(table_p, table.cheader(), 4) == 2);
    REQUIRE(table_p == index_table{{0, 2}, {0}, {2}, {2}});
  }

  SECTION("we can transpose an arbitrary table") {
    index_table table{{0, 1}, {0, 2, 3}, {2, 3, 4}, {0}};
    REQUIRE(transpose(table_p, table.cheader(), 5) == 5);
    REQUIRE(table_p == index_table{{0, 1, 3}, {0}, {1, 2}, {1, 2}, {2}});
  }
}
