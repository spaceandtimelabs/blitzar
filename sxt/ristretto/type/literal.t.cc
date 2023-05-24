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
#include "sxt/ristretto/type/literal.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/property/curve.h"

using namespace sxt;
using namespace sxt::rstt;

TEST_CASE("we can form ristretto points from literals") {
  SECTION("points are on curve21") {
    auto p1 = 0x0_rs;
    REQUIRE(c21p::is_on_curve(p1));
    auto p2 = 0x1_rs;
    REQUIRE(c21p::is_on_curve(p2));
    REQUIRE(p1 != p2);
  }

  SECTION("we handle larger points") {
    auto p1 = 0xffffffffffffffff_rs;
    REQUIRE(c21p::is_on_curve(p1));
    auto p2 = 0x1ffffffffffffffff_rs;
    REQUIRE(c21p::is_on_curve(p2));
    REQUIRE(p1 != p2);
  }

  SECTION("we handle compressed points") {
    auto x = 0x123_crs;
    auto y = 0x123_rs;
    compressed_element y_p;
    rstb::to_bytes(y_p.data(), y);
    REQUIRE(x == y_p);
  }
}
