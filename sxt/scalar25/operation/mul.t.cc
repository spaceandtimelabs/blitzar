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
#include "sxt/scalar25/operation/mul.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt::s25o;
using namespace sxt::s25t;

TEST_CASE("we check the zero value multiplication") {
  element a = 0x2_s25, b = 0x0_s25;

  SECTION("simple multiplication is valid") {
    element s;
    mul(s, a, b);
    REQUIRE(s == 0x0_s25);
  }

  SECTION("commutative property is valid") {
    element s;
    mul(s, b, a);
    REQUIRE(s == 0x0_s25);
  }
}

TEST_CASE("we check that a value times the identity is the same value") {
  element a = 0x2_s25, b = 0x1_s25;
  element expected_s = a;

  SECTION("simple multiplication is valid") {
    element s;
    mul(s, a, b);
    REQUIRE(s == expected_s);
  }

  SECTION("commutative property is valid") {
    element s;
    mul(s, b, a);
    REQUIRE(s == expected_s);
  }
}

TEST_CASE("all inputs and results are smaller than L (L = the field order)") {
  element a = 0x2_s25, b = 0x3_s25;

  SECTION("simple multiplication is valid") {
    element s;
    mul(s, a, b);
    REQUIRE(s == 0x6_s25);
  }

  SECTION("commutative property is valid") {
    element s;
    mul(s, b, a);
    REQUIRE(s == 0x6_s25);
  }
}

TEST_CASE("all input are small than L but result is bigger (L = the field order)") {
  element a = 0x5_s25;
  element b = 0x40000000000000000000000000000000537be77a8bde735960498c6973d74fb_s25; // b = L / 4

  SECTION("simple multiplication is valid") {
    element s;
    mul(s, a, b);
    REQUIRE(s == 0x40000000000000000000000000000000537be77a8bde735960498c6973d74fa_s25);
  }

  SECTION("commutative property is valid") {
    element s;
    mul(s, b, a);
    REQUIRE(s == 0x40000000000000000000000000000000537be77a8bde735960498c6973d74fa_s25);
  }
}

TEST_CASE("some input is bigger than L (L = the field order)") {
  element a = 0x5_s25;
  element b =
      0x2000000000000000000000000000000029bdf3bd45ef39acb024c634b9eba7db_s25; // b = 2 * L + 1

  SECTION("simple multiplication is valid") {
    element s{};
    mul(s, a, b);
    REQUIRE(s == 0x5_s25);
  }

  SECTION("commutative property is valid") {
    element s{};
    mul(s, b, a);
    REQUIRE(s == 0x5_s25);
  }
}

TEST_CASE("all inputs are bigger than L (L = the field order)") {
  element a =
      0x2000000000000000000000000000000029bdf3bd45ef39acb024c634b9eba7db_s25; // a = 2 * L + 1
  element b =
      0x2000000000000000000000000000000029bdf3bd45ef39acb024c634b9eba7dc_s25; // b = 2 * L + 2

  SECTION("simple multiplication is valid") {
    element s;
    mul(s, a, b);
    REQUIRE(s == 0x2_s25);
  }

  SECTION("commutative property is valid") {
    element s{};
    mul(s, b, a);
    REQUIRE(s == 0x2_s25);
  }
}

TEST_CASE("we correctly multiply A * B + C when A, B, C are the biggest 256bits integers") {
  element a = 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_s25;
  element b = 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_s25;

  SECTION("simple multiplication is valid") {
    element s;
    mul(s, a, b);
    REQUIRE(s == 0x399411b7c309a3dceec73d217f5be686bed577bc7792e12a652752ee3568ca2_s25);
  }

  SECTION("commutative property is valid") {
    element s{};
    mul(s, b, a);
    REQUIRE(s == 0x399411b7c309a3dceec73d217f5be686bed577bc7792e12a652752ee3568ca2_s25);
  }
}
