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
#include "sxt/field51/operation/mul.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field51/type/element.h"
#include "sxt/field51/type/literal.h"

using namespace sxt::f51o;
using namespace sxt::f51t;

TEST_CASE("we can multiply finite field elements") {
  auto e1 = 0x711a90c454965634b0962b2b4479551d887ad8d7f33d62f626648de22323dba0_f51;
  auto e2 = 0x48674afb484b050fdcccf508dfb8ce91c364ab4d15584711cba01736e1c59deb_f51;
  element res;

  SECTION("verify against precomputed values") {
    mul(res, e1, e2);
    auto expected_res = 0x6b9ecf8d8ab80bb3a98db6783ec540c9d2fab4684e954e733461f8f187a84ff8_f51;
    REQUIRE(res == expected_res);
  }

  SECTION("we can multiply by 32-bit integers") {
    mul32(res, e1, 0x123u);
    element expected;
    mul(expected, e1, 0x123_f51);
    REQUIRE(res == expected);
  }
}
