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
#include "sxt/base/bit/span_op.h"

#include <cstdint>

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::basbt;

TEST_CASE("we can perform bit operations on spans of bytes") {
  SECTION("we can or_equal two spans") {
    uint64_t x1 = 0b00100011;
    uint64_t x2 = 0b100;
    basct::span<uint8_t> s1{reinterpret_cast<uint8_t*>(&x1), 2};
    basct::cspan<uint8_t> s2{reinterpret_cast<uint8_t*>(&x2), 2};
    or_equal(s1, s2);
    REQUIRE(x1 == 0b00100111);
  }

  SECTION("we can max_equal two spans") {
    uint64_t x1 = 0b00100011;
    uint64_t x2 = 0b100;
    basct::span<uint8_t> s1{reinterpret_cast<uint8_t*>(&x1), 2};
    basct::cspan<uint8_t> s2{reinterpret_cast<uint8_t*>(&x2), 2};
    max_equal(s1, s2);
    REQUIRE(x1 == 0b00100011);
  }

  SECTION("we can max_equal two spans when the rhs is greather") {
    uint64_t x1 = 0b00000011;
    uint64_t x2 = 0b100;
    basct::span<uint8_t> s1{reinterpret_cast<uint8_t*>(&x1), 2};
    basct::cspan<uint8_t> s2{reinterpret_cast<uint8_t*>(&x2), 2};
    max_equal(s1, s2);
    REQUIRE(x1 == 0b00000100);
  }

  SECTION("we can test bits") {
    uint8_t data[] = {0b101, 0b011, 0b1};
    REQUIRE(test_bit(data, 0));
    REQUIRE(!test_bit(data, 1));
    REQUIRE(test_bit(data, 2));
    REQUIRE(!test_bit(data, 3));
    REQUIRE(!test_bit(data, 7));
    REQUIRE(test_bit(data, 8));
    REQUIRE(test_bit(data, 9));
    REQUIRE(!test_bit(data, 10));
    REQUIRE(!test_bit(data, 15));
    REQUIRE(test_bit(data, 16));
    REQUIRE(!test_bit(data, 17));
    REQUIRE(!test_bit(data, 23));
  }
}
