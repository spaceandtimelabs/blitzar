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
#include "sxt/field25/type/element.h"

#include <sstream>

#include "sxt/base/test/unit_test.h"
#include "sxt/field25/base/constants.h"

using namespace sxt;
using namespace sxt::f25t;

TEST_CASE("element conversion") {
  std::ostringstream oss;

  SECTION("of zero prints as zero") {
    element e{0, 0, 0, 0};
    oss << e;
    REQUIRE(oss.str() == "0x0_f25");
  }

  SECTION("of one in Montgomery form prints as one") {
    element e(f25b::r_v.data());
    oss << e;
    REQUIRE(oss.str() == "0x1_f25");
  }

  SECTION("of the modulus prints as zero") {
    element e(f25b::p_v.data());
    oss << e;
    REQUIRE(oss.str() == "0x0_f25");
  }

  SECTION("of r2_v is r_v") {
    element e(f25b::r2_v.data());
    oss << e;
    REQUIRE(oss.str() == "0xe0a77c19a07df2f666ea36f7879462c0a78eb28f5c70b3dd35d438dc58f0d9d_f25");
  }
}
