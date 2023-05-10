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
#include "sxt/scalar25/operation/complement.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt::s25o;
using namespace sxt::s25t;

TEST_CASE("we check that one is the complement of zero") {
  element c;
  complement(c, 0x1_s25); // s = 1 - 1
  REQUIRE(c == 0x0_s25);

  complement(c, c); // s = 1 - 0
  REQUIRE(c == 0x1_s25);

  // 2^252 + 27742317777372353535851937790883648493
  complement(c, 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed_s25);
  REQUIRE(c == 0x1_s25);
}

TEST_CASE("we check that very small values have valid complements") {
  element c;
  complement(c, 0x2_s25); // s = 1 - 2
  REQUIRE(c == 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ec_s25);

  complement(c, c);
  REQUIRE(c == 0x2_s25);

  complement(c, 0x5_s25); // s = 1 - 2
  REQUIRE(c == 0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3e9_s25);

  complement(c, c); // s = 1 - 2
  REQUIRE(c == 0x5_s25);
}

TEST_CASE(
    "we check that a big value (smaller than L) has a valid complement (L = the field order)") {
  element c;
  complement(c, 0x40000000000000000000000000000000537be77a8bde735960498c6973d74fe_s25);
  REQUIRE(c == 0xc0000000000000000000000000000000fa73b66fa39b5a0c20dca53c5b85ef0_s25);

  complement(c, c);
  REQUIRE(c == 0x40000000000000000000000000000000537be77a8bde735960498c6973d74fe_s25);
}

TEST_CASE("we correctly find the complement of A when A is the biggest 256bits integer") {
  element c;
  complement(c, 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_s25);
  REQUIRE(c == 0x14def9dea2f79cd65812631a5cf5d3ed2_s25);

  complement(c, c);
  REQUIRE(c == 0xffffffffffffffffffffffffffffffec6ef5bf4737dcf70d6ec31748d98951c_s25);
}
