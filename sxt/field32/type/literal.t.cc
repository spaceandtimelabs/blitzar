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
#include "sxt/field32/type/literal.h"

#include <sstream>

#include "sxt/base/test/unit_test.h"

using namespace sxt::f32t;

TEST_CASE("literal element printing") {
  std::ostringstream oss;

  SECTION("of zero prints 0x0_f32") {
    oss << 0x0_f32;
    REQUIRE(oss.str() == "0x0_f32");
  }

  SECTION("of one prints 0x1_f32") {
    oss << 0x1_f32;
    REQUIRE(oss.str() == "0x1_f32");
  }

  SECTION("of 10 prints 0xa_f32") {
    oss << 0xa_f32;
    REQUIRE(oss.str() == "0xa_f32");
  }

  SECTION("of 16 prints 0x10_f32") {
    oss << 0x10_f32;
    REQUIRE(oss.str() == "0x10_f32");
  }

  SECTION("of the modulus prints 0x0_f32") {
    oss << 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47_f32;
    REQUIRE(oss.str() == "0x0_f32");
  }

  SECTION("of the modulus minus one prints a pre-computed value") {
    oss << 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd46_f32;
    REQUIRE(oss.str() == "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd46_f32");
  }

  SECTION("of the modulus plus one prints as one") {
    oss << 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd48_f32;
    REQUIRE(oss.str() == "0x1_f32");
  }
}
