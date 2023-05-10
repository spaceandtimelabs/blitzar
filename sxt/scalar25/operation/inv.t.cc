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
#include "sxt/scalar25/operation/inv.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt::s25o;
using namespace sxt::s25t;

TEST_CASE("we check that the identity value has a valid inverse") {
  element s;
  inv(s, 0x1_s25);
  REQUIRE(s == 0x1_s25);
}

TEST_CASE("we check that very small values have valid inverses") {
  element s;
  inv(s, 0x4_s25);
  REQUIRE(s == 0xc0000000000000000000000000000000fa73b66fa39b5a0c20dca53c5b85ef2_s25);

  inv(s, 0x7b_s25);
  REQUIRE(s == 0xa0429a0429a0429a0429a0429a0429a113a6a78a4758bcf0743b4ac99ef39be_s25);
}

TEST_CASE("we check that a big value (smaller than L) has a valid inverse (L = the field order)") {
  element s;
  element a =
      0x40000000000000000000000000000000537be77a8bde735960498c6973d74fe_s25; // a = L / 4 + 3
  inv(s, a);
  REQUIRE(s == 0xa2e8ba2e8ba2e8ba2e8ba2e8ba2e8ba3bd3b647dc11ef7120c5e1f980f986dd_s25);
}

TEST_CASE(
    "we check that a big value (even bigger than L) has a valid inverse (L = the field order)") {
  element s;
  element a =
      0x2000000000000000000000000000000029bdf3bd45ef39acb024c634b9eba7df_s25; // a = 2 * L + 5
  inv(s, a);
  REQUIRE(s == 0x3333333333333333333333333333333375fcb92ed64b8f7ab36e09edf645d96_s25);
}

TEST_CASE("we can do bulk inversion") {
  element sx[2] = {0x123_s25, 0x456_s25};
  element sx_inv[2];
  batch_inv(sx_inv, sx);

  element expected;
  inv(expected, sx[0]);
  REQUIRE(sx_inv[0] == expected);

  inv(expected, sx[1]);
  REQUIRE(sx_inv[1] == expected);
}
