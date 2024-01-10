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
#include "sxt/field25/type/literal.h"

#include <sstream>

#include "sxt/base/test/unit_test.h"

using namespace sxt::f25t;

TEST_CASE("literal element printing") {
  std::ostringstream oss;

  SECTION("of zero prints 0x0_f12") {
    oss << 0x0_f12;
    REQUIRE(oss.str() == "0x0_f12");
  }

  SECTION("of one prints 0x1_f12") {
    oss << 0x1_f12;
    REQUIRE(oss.str() == "0x1_f12");
  }

  SECTION("of 10 prints 0xa_f12") {
    oss << 0xa_f12;
    REQUIRE(oss.str() == "0xa_f12");
  }

  SECTION("of 16 prints 0x10_f12") {
    oss << 0x10_f12;
    REQUIRE(oss.str() == "0x10_f12");
  }

  SECTION("of the modulus prints 0x0_f12") {
    oss << 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab_f12;
    REQUIRE(oss.str() == "0x0_f12");
  }

  SECTION("of the modulus minus one prints a pre-computed value") {
    oss << 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaaa_f12;
    REQUIRE(oss.str() == "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfff"
                         "eb153ffffb9feffffffffaaaa_f12");
  }
}
