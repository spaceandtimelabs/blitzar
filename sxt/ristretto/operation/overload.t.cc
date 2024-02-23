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
#include "sxt/ristretto/operation/overload.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve32/operation/overload.h"
#include "sxt/ristretto/operation/compression.h"
#include "sxt/ristretto/type/literal.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::rstt;
using sxt::s25t::operator""_s25;

TEST_CASE("we can use operators on ristretto elements") {
  auto e1 = 0x123_crs;

  SECTION("we do basic operations") {
    auto e1x2 = e1 + e1;
    REQUIRE(e1x2 != e1);
    REQUIRE(e1x2 == 0x2_s25 * e1);
    REQUIRE(e1 + (e1 - e1) == e1);
    REQUIRE(e1 + (e1 + -e1) == e1);
    REQUIRE(-(-e1) == e1);
  }

  SECTION("we can do operations with a mixture of compressed and uncompressed points") {
    auto e1c = 0x123_crs;
    auto e1 = 0x123_rs;
    auto e2c = 0x456_crs;

    // operator+
    REQUIRE(e1 + e2c == e1c + e2c);
    REQUIRE(e2c + e1 == e1c + e2c);

    // operator-
    REQUIRE(e1 - e2c == e1c - e2c);
    REQUIRE(e2c - e1 == e2c - e1c);
  }

  SECTION("we can use +=") {
    auto e1p = e1;
    e1p += e1;
    REQUIRE(e1p == 0x2_s25 * e1);
  }

  SECTION("we can use -=") {
    auto e1p = e1;
    e1p -= e1;
    REQUIRE(e1p + e1 == e1);
  }
}
