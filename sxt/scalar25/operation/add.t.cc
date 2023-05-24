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
#include "sxt/scalar25/operation/add.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt::s25o;
using namespace sxt::s25t;

TEST_CASE("we check the zero value addition") {
  element a = 0x2_s25, b = 0x0_s25;

  SECTION("simple addition is valid") {
    element s;
    add(s, a, b);
    REQUIRE(s == 0x2_s25);
  }

  SECTION("commutative property is valid") {
    element s;
    add(s, b, a);
    REQUIRE(s == 0x2_s25);
  }
}

TEST_CASE("all inputs and results are smaller than L (L = the field order)") {
  element a = 0x2_s25, b = 0x3_s25;

  SECTION("simple addition is valid") {
    element s;
    add(s, a, b);
    REQUIRE(s == 0x5_s25);
  }

  SECTION("commutative property is valid") {
    element s;
    add(s, b, a);
    REQUIRE(s == 0x5_s25);
  }
}

TEST_CASE("all input are small than L but result is bigger (L = the field order)") {
  element a = 0x5_s25;
  element b = 0x40000000000000000000000000000000537be77a8bde735960498c6973d74fb_s25; // b = L / 4

  SECTION("simple addition is valid") {
    element s;
    add(s, a, b);
    REQUIRE(s == 0x40000000000000000000000000000000537be77a8bde735960498c6973d7500_s25);
  }

  SECTION("commutative property is valid") {
    element s;
    add(s, b, a);
    REQUIRE(s == 0x40000000000000000000000000000000537be77a8bde735960498c6973d7500_s25);
  }
}

TEST_CASE("some input is bigger than L (L = the field order)") {
  element a = 0x5_s25;
  element b =
      0x2000000000000000000000000000000029bdf3bd45ef39acb024c634b9eba7db_s25; // b = 2 * L + 1

  SECTION("simple addition is valid") {
    element s;
    add(s, a, b);
    REQUIRE(s == 0x6_s25);
  }

  SECTION("commutative property is valid") {
    element s;
    add(s, b, a);
    REQUIRE(s == 0x6_s25);
  }
}

TEST_CASE("all inputs are bigger than L (L = the field order)") {
  element a =
      0x2000000000000000000000000000000029bdf3bd45ef39acb024c634b9eba7db_s25; // a = 2 * L + 1
  element b =
      0x2000000000000000000000000000000029bdf3bd45ef39acb024c634b9eba7dc_s25; // b = 2 * L + 2

  SECTION("simple addition is valid") {
    element s;
    add(s, a, b);
    REQUIRE(s == 0x3_s25);
  }

  SECTION("commutative property is valid") {
    element s;
    add(s, b, a);
    REQUIRE(s == 0x3_s25);
  }
}

TEST_CASE("we correctly sum A + B when A is the biggest 256bits integer") {
  element a = 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_s25;
  element b = 0x2_s25;

  SECTION("simple addition is valid") {
    element s;
    add(s, a, b);
    REQUIRE(s == 0xffffffffffffffffffffffffffffffec6ef5bf4737dcf70d6ec31748d98951e_s25);
  }

  SECTION("commutative property is valid") {
    element s;
    add(s, b, a);
    REQUIRE(s == 0xffffffffffffffffffffffffffffffec6ef5bf4737dcf70d6ec31748d98951e_s25);
  }
}

TEST_CASE("we correctly sum A + B when A and B are the biggest 256bits integers") {
  element a = 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_s25;
  element b = 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_s25;

  SECTION("simple addition is valid") {
    element s;
    add(s, a, b);
    REQUIRE(s == 0xffffffffffffffffffffffffffffffd78ffbe0a4404020b55c5ffcebe3b564b_s25);
  }

  SECTION("commutative property is valid") {
    element s;
    add(s, b, a);
    REQUIRE(s == 0xffffffffffffffffffffffffffffffd78ffbe0a4404020b55c5ffcebe3b564b_s25);
  }
}
