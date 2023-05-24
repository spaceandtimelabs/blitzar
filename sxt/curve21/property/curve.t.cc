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
#include "sxt/curve21/property/curve.h"

#include <iostream>

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/type/literal.h"

using namespace sxt;
using namespace sxt::c21p;
using c21t::operator""_c21;

TEST_CASE("we can determine if a point is on the curve") {
  auto p = 0x1_c21;
  REQUIRE(is_on_curve(p));

  auto p2 = p;
  p2.X[0] += 1;
  REQUIRE(!is_on_curve(p2));
}
