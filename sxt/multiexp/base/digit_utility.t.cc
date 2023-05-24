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
#include "sxt/multiexp/base/digit_utility.h"

#include <array>

#include "sxt/base/test/unit_test.h"

using namespace sxt::mtxb;

TEST_CASE("we can extract digits given a radix that's a power of two") {
  SECTION("we can extract digits from a byte using a radix of 2^8") {
    uint8_t val[] = {0b10};
    uint8_t digit[1];
    extract_digit(digit, val, 8, 0);
    REQUIRE(digit[0] == 0b10);
  }

  SECTION("we can extract digits from a byte with a radix less than 2^8") {
    uint8_t val[] = {0b10110011};
    uint8_t digit[1];

    extract_digit(digit, val, 3, 0);
    REQUIRE(digit[0] == 0b011);

    extract_digit(digit, val, 3, 1);
    REQUIRE(digit[0] == 0b110);

    extract_digit(digit, val, 3, 2);
    REQUIRE(digit[0] == 0b010);
  }

  SECTION("we can extract digits from a multi-byte number") {
    uint8_t val[] = {0b10110011, 0b11001011};
    uint8_t digit[1];

    extract_digit(digit, val, 8, 0);
    REQUIRE(digit[0] == val[0]);

    extract_digit(digit, val, 8, 1);
    REQUIRE(digit[0] == val[1]);

    extract_digit(digit, val, 3, 0);
    REQUIRE(digit[0] == 0b011);

    extract_digit(digit, val, 3, 1);
    REQUIRE(digit[0] == 0b110);

    extract_digit(digit, val, 3, 2);
    REQUIRE(digit[0] == 0b110);

    extract_digit(digit, val, 3, 3);
    REQUIRE(digit[0] == 0b101);

    extract_digit(digit, val, 3, 4);
    REQUIRE(digit[0] == 0b100);

    extract_digit(digit, val, 3, 5);
    REQUIRE(digit[0] == 0b001);
  }

  SECTION("we can extract digits of size greater than a byte") {
    uint8_t val[] = {0b10110011, 0b11001011};
    uint8_t digit[2];

    extract_digit(digit, val, 9, 0);
    REQUIRE(digit[0] == val[0]);
    REQUIRE(digit[1] == 0b1);

    extract_digit(digit, val, 9, 1);
    REQUIRE(digit[0] == 0b1100101);
    REQUIRE(digit[1] == 0);
  }

  SECTION("we can exract a digit of the maximum size") {
    uint8_t val[] = {0b10110011, 0b11001011};
    uint8_t digit[2];

    extract_digit(digit, val, 16, 0);
    REQUIRE(digit[0] == val[0]);
    REQUIRE(digit[1] == val[1]);
  }
}

TEST_CASE("we can determine if a digit is zero") {
  SECTION("we handle a zero exponent of a single byte") {
    uint8_t val[] = {0};
    REQUIRE(is_digit_zero(val, 8, 0));
    REQUIRE(is_digit_zero(val, 1, 0));
    REQUIRE(is_digit_zero(val, 1, 1));
    REQUIRE(is_digit_zero(val, 7, 1));
    REQUIRE(is_digit_zero(val, 9, 0));
  }

  SECTION("we handle a single byte exponent of 1") {
    uint8_t val[] = {1};
    REQUIRE(!is_digit_zero(val, 8, 0));
    REQUIRE(!is_digit_zero(val, 7, 0));
    REQUIRE(is_digit_zero(val, 7, 1));
    REQUIRE(!is_digit_zero(val, 9, 0));
  }

  SECTION("we handle two bytes") {
    uint8_t val[] = {0, 1};
    REQUIRE(!is_digit_zero(val, 9, 0));
    REQUIRE(is_digit_zero(val, 9, 1));

    val[1] = 0b10000000;
    REQUIRE(!is_digit_zero(val, 16, 0));
    REQUIRE(is_digit_zero(val, 15, 0));
  }
}

TEST_CASE("we can count the number of digits in a number") {
  SECTION("we correctly count zero digits") {
    REQUIRE(count_num_digits(std::array<uint8_t, 1>{0}, 2) == 0);
  }

  SECTION("we correctly count a single digit") {
    REQUIRE(count_num_digits(std::array<uint8_t, 1>{1}, 2) == 1);
  }

  SECTION("we correctly count the maximum number of digits") {
    REQUIRE(count_num_digits(std::array<uint8_t, 1>{255}, 2) == 4);
  }
}
