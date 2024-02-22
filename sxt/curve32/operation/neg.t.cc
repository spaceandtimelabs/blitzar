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
#include "sxt/curve32/operation/neg.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve32/operation/add.h"
#include "sxt/curve32/type/literal.h"

using namespace sxt;
using namespace sxt::c32o;
using c32t::operator""_c32;

TEST_CASE("we can negate curve-21 elements") {
  SECTION("properties of negatives are satisfied") {
    auto p = 0x123_c32;
    c32t::element_p3 np;
    neg(np, p);
    c32t::element_p3 z;
    add(z, p, np);
    auto q = 0x456_c32;
    c32t::element_p3 qp;
    add(qp, q, z);
    REQUIRE(q == qp);
  }

  SECTION("we can conditionally negate elements") {
    auto p = 0x123_c32;
    c32t::element_p3 np;
    neg(np, p);
    c32t::element_p3 z = p;
    cneg(z, 1);
    REQUIRE(z == np);
    z = p;
    cneg(z, 0);
    REQUIRE(z == p);
  }
}
