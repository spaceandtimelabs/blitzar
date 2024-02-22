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
#include "sxt/field51/operation/abs.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field51/type/element.h"
#include "sxt/field51/type/literal.h"

using namespace sxt;
using namespace sxt::f51o;
using namespace sxt::f51t;

TEST_CASE("abs") {
  SECTION("negative element becomes positive") {
    auto e = 0x48674afb484b050fdcccf508dfb8ce91c364ab4d15584711cba01736e1c59deb_f51;
    f51t::element res;
    abs(res, e);
    auto expected_res = 0x3798b504b7b4faf023330af72047316e3c9b54b2eaa7b8ee345fe8c91e3a6202_f51;
    REQUIRE(res == expected_res);
  }

  SECTION("positive element remains positive") {
    auto e = 0x711a90c454965634b0962b2b4479551d887ad8d7f33d62f626648de22323dba0_f51;
    f51t::element res;
    abs(res, e);
    REQUIRE(res == e);
  }
}
