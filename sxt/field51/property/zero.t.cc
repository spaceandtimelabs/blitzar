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
#include "sxt/field51/property/zero.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field51/type/literal.h"

using namespace sxt::f51t;
using namespace sxt::f51p;

TEST_CASE("we can determine if a field element is zero") {
  REQUIRE(is_zero(0x0_f51));
  REQUIRE(!is_zero(0x1_f51));
  REQUIRE(is_zero(0x0_f51));

  // e = 2^255 - 19
  auto e = 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffec_f51;
  e[0] += 1;

  REQUIRE(is_zero(e));
}
