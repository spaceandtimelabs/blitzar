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
#include "sxt/base/num/abs.h"

#include <cstdint>
#include <limits>

#include "sxt/base/test/unit_test.h"
#include "sxt/base/type/int.h"

using namespace sxt;
using namespace sxt::basn;

TEST_CASE("we can compute the absolute value of numbers") {
  SECTION("we can compute the absolute value of numbers up to 8 bytes") {
    REQUIRE(abs(0) == 0);
    REQUIRE(abs(1) == 1);
    REQUIRE(abs(-1) == 1);
    REQUIRE(abs(-1ll) == 1ll);
  }

  SECTION("we can compute the absolute value of numbers larger than 8 bytes") {
    REQUIRE(abs(int128_t{-1}) == 1);
    REQUIRE(abs(int128_t{1}) == 1);
    REQUIRE(abs(int128_t{-2}) == 2);
    REQUIRE(abs(int128_t{2}) == 2);
  }
}

TEST_CASE("we can take the absolute value of a number and convert to unsigned") {
  SECTION("we handle some basic examples") {
    REQUIRE(abs_to_unsigned(0) == 0u);
    REQUIRE(abs_to_unsigned(1) == 1u);
    REQUIRE(abs_to_unsigned(-1) == 1u);
    REQUIRE(abs_to_unsigned(-1ll) == 1ull);
  }

  SECTION("we handle 128 bit numbers") {
    REQUIRE(abs_to_unsigned(int128_t{-1}) == 1);
    REQUIRE(abs_to_unsigned(int128_t{1}) == 1);
    REQUIRE(abs_to_unsigned(int128_t{-2}) == 2);
    REQUIRE(abs_to_unsigned(int128_t{2}) == 2);
  }

  SECTION("we handle extreme values") {
    REQUIRE(abs_to_unsigned(std::numeric_limits<int64_t>::min()) == 9'223'372'036'854'775'808ull);
    REQUIRE(abs_to_unsigned(std::numeric_limits<int64_t>::max()) == 9'223'372'036'854'775'807ull);
  }
}
