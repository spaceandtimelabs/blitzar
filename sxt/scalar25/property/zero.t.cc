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
#include "sxt/scalar25/property/zero.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt::s25t;
using namespace sxt::s25p;

TEST_CASE("we can determine if a field element is zero") {
  REQUIRE(is_zero(element{}));
  REQUIRE(is_zero(element{0u}));
  REQUIRE(!is_zero(element{1u}));
  REQUIRE(is_zero(element{0u}));

  // e = 2^252 + 27742317777372353535851937790883648493
  REQUIRE(is_zero(0x1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed_s25));
}
