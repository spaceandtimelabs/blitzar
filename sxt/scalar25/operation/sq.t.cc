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
#include "sxt/scalar25/operation/sq.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt::s25o;
using namespace sxt::s25t;

TEST_CASE("we check the zero squared case (s = 0 * 0)") {
  element s = 0x0_s25;
  sq(s, 0x0_s25);
  REQUIRE(s == 0x0_s25);
}

TEST_CASE("we check the one squared case (s = 1 * 1)") {
  element s = 0x0_s25;
  sq(s, 0x1_s25);
  REQUIRE(s == 0x1_s25);
}

TEST_CASE("we check the result of small values squared") {
  element s = 0x0_s25;
  sq(s, 0x2_s25);
  REQUIRE(s == 0x4_s25);

  s = 0x0_s25;
  sq(s, 0x3_s25);
  REQUIRE(s == 0x9_s25);
}

TEST_CASE("we check that a big value, smaller than L, can generate a valid reduced value (L = the "
          "field order)") {
  element s = 0x0_s25;
  element a = 0x40000000000000000000000000000000537be77a8bde735960498c6973d74fb_s25; // a = L / 4
  sq(s, a);
  REQUIRE(s == 0xb0000000000000000000000000000000e594bc9100a3bd35c8ca4221fe901b3_s25);
}

TEST_CASE("we check that a value bigger than L can generate a valid reduced value (L = the field "
          "order)") {
  element s = 0x0_s25;
  element a =
      0x2000000000000000000000000000000029bdf3bd45ef39acb024c634b9eba7dd_s25; // a = 2 * L + 3
  sq(s, a);
  REQUIRE(s == 0x9_s25);
}
