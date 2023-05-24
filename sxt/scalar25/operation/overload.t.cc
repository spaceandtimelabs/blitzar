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
#include "sxt/scalar25/operation/overload.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::s25t;

TEST_CASE("we can use operators on scalars") {
  auto x = 0x1_s25;

  SECTION("we can use basic operations") {
    REQUIRE(0x1_s25 + 0x2_s25 == 0x3_s25);
    REQUIRE(0x2_s25 - 0x1_s25 == 0x1_s25);
    REQUIRE(0x2_s25 * 0x3_s25 == 0x6_s25);
    REQUIRE(-0x2_s25 == 0x0_s25 - 0x2_s25);
    REQUIRE(0x6_s25 / 0x2_s25 == 0x3_s25);
  }

  SECTION("we can use +=") {
    x += 0x2_s25;
    REQUIRE(x == 0x3_s25);
  }

  SECTION("we can use -=") {
    x -= 0x2_s25;
    REQUIRE(x == -0x1_s25);
  }

  SECTION("we can use *=") {
    x *= 0x2_s25;
    REQUIRE(x == 0x2_s25);
  }

  SECTION("we can use /=") {
    x /= 0x2_s25;
    REQUIRE(x * 0x2_s25 == 0x1_s25);
  }
}
