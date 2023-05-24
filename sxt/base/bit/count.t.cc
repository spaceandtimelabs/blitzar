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
#include "sxt/base/bit/count.h"

#include <limits>

#include "sxt/base/test/unit_test.h"

using namespace sxt::basbt;

TEST_CASE("we can count the number of leading zeros in a blob of data") {
  SECTION("we correctly handle the case when there are no bits set") {
    uint8_t data1[1] = {};
    REQUIRE(count_leading_zeros(data1, sizeof(data1)) == 8);

    uint64_t data2[2] = {};
    REQUIRE(count_leading_zeros(reinterpret_cast<uint8_t*>(data2), sizeof(data2)) == 128);
  }

  SECTION("we correctly identify the number of leading zeros when there are bits "
          "set") {
    uint8_t data1[1] = {0b1};
    REQUIRE(count_leading_zeros(data1, sizeof(data1)) == 7);

    data1[0] = 0b10;
    REQUIRE(count_leading_zeros(data1, sizeof(data1)) == 6);

    data1[0] = 0b10000000;
    REQUIRE(count_leading_zeros(data1, sizeof(data1)) == 0);

    uint64_t data2[2] = {0b1};
    REQUIRE(count_leading_zeros(reinterpret_cast<uint8_t*>(data2), sizeof(data2)) == 127);

    data2[0] = 0b10;
    REQUIRE(count_leading_zeros(reinterpret_cast<uint8_t*>(data2), sizeof(data2)) == 126);

    data2[0] = std::numeric_limits<uint64_t>::max();
    REQUIRE(count_leading_zeros(reinterpret_cast<uint8_t*>(data2), sizeof(data2)) == 64);

    data2[1] = 0b1;
    REQUIRE(count_leading_zeros(reinterpret_cast<uint8_t*>(data2), sizeof(data2)) == 63);

    data2[1] = 0b10;
    REQUIRE(count_leading_zeros(reinterpret_cast<uint8_t*>(data2), sizeof(data2)) == 62);

    data2[1] = std::numeric_limits<uint64_t>::max();
    REQUIRE(count_leading_zeros(reinterpret_cast<uint8_t*>(data2), sizeof(data2)) == 0);
  }
}

TEST_CASE("we can count the number of 1 bits set in a blob of data") {
  SECTION("we count zero when there are no 1 bits") {
    uint8_t data[] = {0, 0};
    REQUIRE(pop_count(data, sizeof(data)) == 0);
  }

  SECTION("we count 1 bits") {
    uint8_t data[] = {0, 0b101};
    REQUIRE(pop_count(data, sizeof(data)) == 2);
  }

  SECTION("we count 1 bits in data segments greater than 64 bits") {
    uint64_t data[] = {0b1, 0b11, 0};
    REQUIRE(pop_count(reinterpret_cast<uint8_t*>(data), sizeof(data)) == 3);
  }
}
