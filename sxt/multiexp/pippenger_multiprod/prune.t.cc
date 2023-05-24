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
#include "sxt/multiexp/pippenger_multiprod/prune.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/multiexp/index/index_table.h"

using namespace sxt;
using namespace sxt::mtxpmp;

TEST_CASE("we can prune the multi-product table to deactive inputs and outputs") {
  SECTION("we handle the empty case") {
    mtxi::index_table table;
    std::vector<uint64_t> markers;
    size_t num_inactive_outputs = 0;
    size_t num_inactive_inputs = 0;
    prune_rows(table.header(), markers, num_inactive_outputs, num_inactive_inputs);

    mtxi::index_table expected_table;
    REQUIRE(table == expected_table);
    REQUIRE(num_inactive_outputs == 0);
    REQUIRE(num_inactive_inputs == 0);
  }

  SECTION("we handle the case when there are no active outputs") {
    mtxi::index_table table{{0, 1, 0}};
    std::vector<uint64_t> markers;
    size_t num_inactive_outputs = 1;
    size_t num_inactive_inputs = 1;
    prune_rows(table.header(), markers, num_inactive_outputs, num_inactive_inputs);

    mtxi::index_table expected_table{{0, 1, 0}};
    REQUIRE(table == expected_table);
    REQUIRE(num_inactive_outputs == 1);
    REQUIRE(num_inactive_inputs == 1);
  }

  SECTION("we handle the case of one output and one deactivation") {
    mtxi::index_table table{{0, 0, 0}};
    std::vector<uint64_t> markers = {99};
    size_t num_inactive_outputs = 0;
    size_t num_inactive_inputs = 0;
    prune_rows(table.header(), markers, num_inactive_outputs, num_inactive_inputs);

    mtxi::index_table expected_table{{0, 1, 0}};
    REQUIRE(table == expected_table);
    REQUIRE(num_inactive_outputs == 1);
    REQUIRE(num_inactive_inputs == 1);
    std::vector<uint64_t> expected_markers = {99};
    REQUIRE(markers == expected_markers);
  }

  SECTION("we handle the case of two outputs, one deactivation, and one "
          "non-deactivation") {
    mtxi::index_table table{{0, 0, 0, 1}, {1, 0, 0}};
    std::vector<uint64_t> markers = {99, 101};
    size_t num_inactive_outputs = 0;
    size_t num_inactive_inputs = 0;
    prune_rows(table.header(), markers, num_inactive_outputs, num_inactive_inputs);

    mtxi::index_table expected_table{{0, 1, 0, 0}, {1, 0, 0}};
    REQUIRE(table == expected_table);
    REQUIRE(num_inactive_outputs == 0);
    REQUIRE(num_inactive_inputs == 1);
    std::vector<uint64_t> expected_markers = {101, 99};
    REQUIRE(markers == expected_markers);
  }

  SECTION("we can deactivate many entries") {
    mtxi::index_table table{{0, 1, 0, 0, 1, 2, 3}, {1, 0, 1}};
    std::vector<uint64_t> markers = {99, 100, 101, 102};
    size_t num_inactive_outputs = 0;
    size_t num_inactive_inputs = 1;
    prune_rows(table.header(), markers, num_inactive_outputs, num_inactive_inputs);

    mtxi::index_table expected_table{{0, 4, 0, 1, 2, 3, 0}, {1, 0, 0}};
    REQUIRE(table == expected_table);
    REQUIRE(num_inactive_outputs == 0);
    REQUIRE(num_inactive_inputs == 4);
    std::vector<uint64_t> expected_markers = {99, 101, 102, 100};
    REQUIRE(markers == expected_markers);
  }

  SECTION("we deactive entries that are only in rows with a single active entry") {
    mtxi::index_table table{{0, 1, 0, 0}, {1, 0, 0}};
    std::vector<uint64_t> markers = {99};
    size_t num_inactive_outputs = 0;
    size_t num_inactive_inputs = 1;
    prune_rows(table.header(), markers, num_inactive_outputs, num_inactive_inputs);

    mtxi::index_table expected_table{{0, 2, 0, 1}, {1, 1, 1}};
    REQUIRE(table == expected_table);
    REQUIRE(num_inactive_outputs == 2);
    REQUIRE(num_inactive_inputs == 2);
    std::vector<uint64_t> expected_markers = {99};
    REQUIRE(markers == expected_markers);
  }

  SECTION("we can handle multiple active entries") {
    mtxi::index_table table{{0, 0, 0, 1, 2}, {1, 0, 0, 2}};
    std::vector<uint64_t> markers = {101, 102, 103};
    size_t num_inactive_outputs = 0;
    size_t num_inactive_inputs = 0;
    prune_rows(table.header(), markers, num_inactive_outputs, num_inactive_inputs);

    mtxi::index_table expected_table{{0, 1, 0, 0, 1}, {1, 0, 0, 1}};
    REQUIRE(table == expected_table);
    std::vector<uint64_t> expected_markers = {102, 101, 103};
    REQUIRE(markers == expected_markers);
  }
}
