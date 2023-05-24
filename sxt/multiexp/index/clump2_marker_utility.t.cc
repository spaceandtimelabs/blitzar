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
#include "sxt/multiexp/index/clump2_marker_utility.h"

#include <iostream>
#include <random>

#include "sxt/base/test/unit_test.h"
#include "sxt/multiexp/index/clump2_descriptor.h"
#include "sxt/multiexp/index/clump2_descriptor_utility.h"
#include "sxt/multiexp/index/random_clump2.h"

using namespace sxt;
using namespace sxt::mtxi;

TEST_CASE("we can convert between a clumped index set and its marker") {
  clump2_descriptor descriptor;

  uint64_t marker, clump_index, index1, index2;

  SECTION("verify conversions for a clump size of 2") {
    init_clump2_descriptor(descriptor, 2);

    marker = compute_clump2_marker(descriptor, 0, 0, 0);
    REQUIRE(marker == 0);
    unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
    REQUIRE(clump_index == 0);
    REQUIRE(index1 == 0);
    REQUIRE(index2 == 0);

    marker = compute_clump2_marker(descriptor, 0, 0, 1);
    REQUIRE(marker == 1);
    unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
    REQUIRE(clump_index == 0);
    REQUIRE(index1 == 0);
    REQUIRE(index2 == 1);

    marker = compute_clump2_marker(descriptor, 0, 1, 1);
    REQUIRE(marker == 2);
    unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
    REQUIRE(clump_index == 0);
    REQUIRE(index1 == 1);
    REQUIRE(index2 == 1);

    marker = compute_clump2_marker(descriptor, 10, 1, 1);
    REQUIRE(marker == 10 * 3 + 2);
    unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
    REQUIRE(clump_index == 10);
    REQUIRE(index1 == 1);
    REQUIRE(index2 == 1);
  }

  SECTION("verify conversions for a clump size of 3") {
    init_clump2_descriptor(descriptor, 3);

    marker = compute_clump2_marker(descriptor, 0, 0, 0);
    REQUIRE(marker == 0);
    unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
    REQUIRE(clump_index == 0);
    REQUIRE(index1 == 0);
    REQUIRE(index2 == 0);

    marker = compute_clump2_marker(descriptor, 0, 0, 1);
    REQUIRE(marker == 1);
    unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
    REQUIRE(clump_index == 0);
    REQUIRE(index1 == 0);
    REQUIRE(index2 == 1);

    marker = compute_clump2_marker(descriptor, 0, 0, 2);
    REQUIRE(marker == 2);
    unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
    REQUIRE(clump_index == 0);
    REQUIRE(index1 == 0);
    REQUIRE(index2 == 2);

    marker = compute_clump2_marker(descriptor, 0, 1, 1);
    REQUIRE(marker == 3);
    unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
    REQUIRE(clump_index == 0);
    REQUIRE(index1 == 1);
    REQUIRE(index2 == 1);

    marker = compute_clump2_marker(descriptor, 0, 1, 2);
    REQUIRE(marker == 4);
    unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
    REQUIRE(clump_index == 0);
    REQUIRE(index1 == 1);
    REQUIRE(index2 == 2);

    marker = compute_clump2_marker(descriptor, 0, 2, 2);
    REQUIRE(marker == 5);
    unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
    REQUIRE(clump_index == 0);
    REQUIRE(index1 == 2);
    REQUIRE(index2 == 2);
  }

  SECTION("verify we can recover indexes from the marker of a random clump") {
    std::mt19937 rng{0};
    for (int i = 0; i < 10; ++i) {
      random_clump2 clump;
      generate_random_clump2(clump, rng);
      init_clump2_descriptor(descriptor, clump.clump_size);

      // 1 subset case
      marker = compute_clump2_marker(descriptor, clump.clump_index, clump.index1, clump.index1);
      unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
      REQUIRE(clump_index == clump.clump_index);
      REQUIRE(index1 == clump.index1);
      REQUIRE(index2 == clump.index1);

      // 2 subset case
      marker = compute_clump2_marker(descriptor, clump.clump_index, clump.index1, clump.index2);
      unpack_clump2_marker(clump_index, index1, index2, descriptor, marker);
      REQUIRE(clump_index == clump.clump_index);
      REQUIRE(index1 == clump.index1);
      REQUIRE(index2 == clump.index2);
    }
  }
}

TEST_CASE("we can consume clump2 markers from a span of indexes") {
  const uint64_t clump_size = 15;
  basct::span<uint64_t> values;
  uint64_t marker;
  clump2_descriptor descriptor;
  init_clump2_descriptor(descriptor, clump_size);

  SECTION("we can consume a single marker") {
    uint64_t data[1] = {0};
    values = data;

    marker = consume_clump2_marker(values, descriptor);
    REQUIRE(marker == 0);
    REQUIRE(values.empty());
  }

  SECTION("we can consume 2 values belonging to the same clump") {
    uint64_t data[2] = {0, clump_size - 1};
    values = data;
    marker = consume_clump2_marker(values, descriptor);
    REQUIRE(marker == compute_clump2_marker(descriptor, 0, data[0], data[1]));
    REQUIRE(values.empty());
  }

  SECTION("we only consume a single value if the next value belongs to a different "
          "clump") {
    uint64_t data[2] = {0, clump_size};
    values = data;
    marker = consume_clump2_marker(values, descriptor);
    REQUIRE(marker == compute_clump2_marker(descriptor, 0, data[0], data[0]));
    REQUIRE(values.size() == 1);
    REQUIRE(&values[0] == &data[1]);
  }
}
