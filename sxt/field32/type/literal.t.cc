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
#include "sxt/base/type/literal.h"

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

  SECTION("random field element prints as expected") {
    oss << 0x3b86191f4f2865cc462f08daa6d911c0df283b53cb3b8f7d6027666f4c94e38_f32;
    REQUIRE(oss.str() == "0x3b86191f4f2865cc462f08daa6d911c0df283b53cb3b8f7d6027666f4c94e38_f32");
  }

  SECTION("modulus prints as 0x0_f32") {
    oss << 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed_f32;
    REQUIRE(oss.str() == "0x0_f32");
  }
}

TEST_CASE("element equality") {
  element f = 0x1_f32;
  REQUIRE(f == f);
  REQUIRE(f == 0x1_f32);
}
