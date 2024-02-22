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
#include "sxt/field32/property/sign.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field32/type/literal.h"

using namespace sxt::f32t;
using namespace sxt::f32p;

TEST_CASE("we can determine if a field element is negative") {
  REQUIRE(is_negative(0x0_f32) == 0);
  REQUIRE(is_negative(0x1_f32) == 1);
  REQUIRE(is_negative(0x2_f32) == 0);

  // e = 2^255 - 19
  auto e = 0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffec_f32;
  e[0] -= 1;
  REQUIRE(is_negative(e) == 1);
  e[0] -= 1;
  REQUIRE(is_negative(e) == 0);
}
