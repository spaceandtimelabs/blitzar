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
#include "sxt/field51/operation/sq.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field51/type/element.h"
#include "sxt/field51/type/literal.h"

using namespace sxt;
using namespace sxt::f51o;
using namespace sxt::f51t;

TEST_CASE("sq") {
  SECTION("regular") {
    auto e = 0x48674afb484b050fdcccf508dfb8ce91c364ab4d15584711cba01736e1c59deb_f51;
    f51t::element res;
    sq(res, e);
    auto expected_res = 0x7fa13403b69cc40197d157d218f6f8afdfe95bc7e98ef46112480fe346aa6ec3_f51;
    REQUIRE(res == expected_res);
  }

  SECTION("times two") {
    auto e = 0x711a90c454965634b0962b2b4479551d887ad8d7f33d62f626648de22323dba0_f51;
    f51t::element res;
    sq2(res, e);
    auto expected_res = 0x55a6d8f01f8a9a9a2385a64a8d3aeae2c0d8895a8027b9fc8725cce6360e0f2_f51;
    REQUIRE(res == expected_res);
  }
}
