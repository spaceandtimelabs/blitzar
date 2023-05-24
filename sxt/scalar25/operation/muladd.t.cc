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
#include "sxt/scalar25/operation/muladd.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt::s25o;
using namespace sxt::s25t;

TEST_CASE("we check the zero value yields valid results") {
  element a = 0x2_s25, b = 0x0_s25;

  SECTION("simple multiplication is valid") {
    element s;
    muladd(s, a, b, a);
    REQUIRE(s == a);
  }

  SECTION("commutative property is valid") {
    element s;
    muladd(s, b, a, b);
    REQUIRE(s == b);
  }
}

TEST_CASE("we check the identity value yields valid results") {
  element a = 0x2_s25, b = 0x1_s25;

  SECTION("simple multiplication is valid") {
    element s;
    muladd(s, a, b, 0x0_s25);
    REQUIRE(s == a);
  }

  SECTION("commutative property is valid") {
    element s;
    muladd(s, b, a, 0x0_s25);
    REQUIRE(s == a);
  }
}

TEST_CASE("all inputs and results are small than L (the field order)") {
  element a = 0x2_s25, b = 0x3_s25, c = 0x7_s25, expected_s = 0xd_s25;

  SECTION("simple multiplication is valid") {
    element s;
    muladd(s, a, b, c);
    REQUIRE(s == expected_s);
  }

  SECTION("commutative property is valid") {
    element s;
    muladd(s, b, a, c);
    REQUIRE(s == expected_s);
  }
}

TEST_CASE("all inputs are smaller than L but result is big (L = the field order)") {
  element a = 0x5_s25;
  element b = 0x40000000000000000000000000000000537be77a8bde735960498c6973d74fb_s25; // b = L / 4
  element c = 0x7_s25;

  SECTION("simple multiplication is valid") {
    element s;
    muladd(s, a, b, c);
    REQUIRE(s == 0x40000000000000000000000000000000537be77a8bde735960498c6973d7501_s25);
  }

  SECTION("commutative property is valid") {
    element s;
    muladd(s, b, a, c);
    REQUIRE(s == 0x40000000000000000000000000000000537be77a8bde735960498c6973d7501_s25);
  }

  SECTION("big addition is valid") {
    element s;
    muladd(s, c, a, b);
    REQUIRE(s == 0x40000000000000000000000000000000537be77a8bde735960498c6973d751e_s25);

    muladd(s, a, c, b);
    REQUIRE(s == 0x40000000000000000000000000000000537be77a8bde735960498c6973d751e_s25);
  }
}

TEST_CASE("some input are bigger than L (L = the field order)") {
  element a = 0x5_s25;
  element b =
      0x2000000000000000000000000000000029bdf3bd45ef39acb024c634b9eba7db_s25; // b = 2 * L + 1
  element c = 0x7_s25;

  SECTION("simple multiplication is valid") {
    element s;
    muladd(s, a, b, c);
    REQUIRE(s == 0xc_s25);
  }

  SECTION("commutative property is valid") {
    element s;
    muladd(s, b, a, c);
    REQUIRE(s == 0xc_s25);
  }

  SECTION("big addition is valid") {
    element s;
    muladd(s, c, a, b);
    REQUIRE(s == 0x24_s25);
  }
}

TEST_CASE("all inputs are bigger than L (L = the field order)") {
  element a =
      0x2000000000000000000000000000000029bdf3bd45ef39acb024c634b9eba7db_s25; // a = 2 * L + 1
  element b =
      0x2000000000000000000000000000000029bdf3bd45ef39acb024c634b9eba7dc_s25; // b = 2 * L + 2
  element c =
      0x2000000000000000000000000000000029bdf3bd45ef39acb024c634b9eba7dd_s25; // c = 2 * L + 3

  SECTION("simple multiplication is valid") {
    element s;
    muladd(s, a, b, c);
    REQUIRE(s == 0x5_s25);
  }

  SECTION("commutative property is valid") {
    element s;
    muladd(s, b, a, c);
    REQUIRE(s == 0x5_s25);
  }
}

TEST_CASE("we correctly multiply A * B + C when A, B, C are the biggest 256bits integers") {
  element a = 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_s25;
  element b = 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_s25;
  element c = 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_s25;

  SECTION("simple multiplication is valid") {
    element s;
    muladd(s, a, b, c);
    REQUIRE(s == 0x399411b7c309a3dceec73d217f5be671dfdb99197ff60ad252c438913f94dd1_s25);
  }

  SECTION("commutative property is valid") {
    element s;
    muladd(s, b, a, c);
    REQUIRE(s == 0x399411b7c309a3dceec73d217f5be671dfdb99197ff60ad252c438913f94dd1_s25);
  }
}
