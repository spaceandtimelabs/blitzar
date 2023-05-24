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
#include "sxt/base/bit/iteration.h"

#include <vector>

#include "sxt/base/test/unit_test.h"

using namespace sxt::basbt;

TEST_CASE("we can iterate through the bits that are set in an integer") {
  uint64_t bitset;

  SECTION("we can iterate over a number with only a single bit set") {
    bitset = 1;
    REQUIRE(consume_next_bit(bitset) == 0);
    REQUIRE(bitset == 0);

    bitset = 1 << 1;
    REQUIRE(consume_next_bit(bitset) == 1);
    REQUIRE(bitset == 0);

    bitset = 1 << 2;
    REQUIRE(consume_next_bit(bitset) == 2);
    REQUIRE(bitset == 0);
  }

  SECTION("we can iterate through a number with two bits set") {
    bitset = 1 | (1 << 1);
    REQUIRE(consume_next_bit(bitset) == 0);
    REQUIRE(bitset != 0);
    REQUIRE(consume_next_bit(bitset) == 1);
    REQUIRE(bitset == 0);

    bitset = 1 | (1 << 2);
    REQUIRE(consume_next_bit(bitset) == 0);
    REQUIRE(bitset != 0);
    REQUIRE(consume_next_bit(bitset) == 2);
    REQUIRE(bitset == 0);

    bitset = (1 << 3) | (1 << 7);
    REQUIRE(consume_next_bit(bitset) == 3);
    REQUIRE(bitset != 0);
    REQUIRE(consume_next_bit(bitset) == 7);
    REQUIRE(bitset == 0);
  }
}

TEST_CASE("we can iterate over the bits in a blob") {
  std::vector<size_t> indexes;
  auto f = [&](size_t index) noexcept { indexes.push_back(index); };

  SECTION("we handle the empty case of a single byte") {
    uint8_t bytes[] = {0};
    for_each_bit(bytes, sizeof(bytes), f);
    REQUIRE(indexes.empty());
  }

  SECTION("we handle the empty case of many bytes") {
    uint8_t bytes[100] = {};
    for_each_bit(bytes, sizeof(bytes), f);
    REQUIRE(indexes.empty());
  }

  SECTION("we handle the case of a single bit") {
    uint8_t bytes[100] = {};
    bytes[0] = 1;
    for_each_bit(bytes, sizeof(bytes), f);
    REQUIRE(indexes == std::vector<size_t>{0});
  }

  SECTION("we handle the case of a single bit at the end") {
    uint8_t bytes[100] = {};
    bytes[99] = 0b10000000;
    for_each_bit(bytes, sizeof(bytes), f);
    REQUIRE(indexes == std::vector<size_t>{799});
  }

  SECTION("we handle multiple bits at various locations") {
    uint8_t bytes[100] = {};
    bytes[0] = 0b100;
    bytes[10] = 0b1;
    bytes[99] = 0b10000000;
    for_each_bit(bytes, sizeof(bytes), f);
    REQUIRE(indexes == std::vector<size_t>{2, 80, 799});
  }
}
