/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2025-present Space and Time Labs, Inc.
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
#include "sxt/proof/sumcheck/polynomial_mapper.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/realization/field.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfsk;
using s25t::operator""_s25;

using T = s25t::element;

TEST_CASE("we can map indexes to expanded polynomials") {
  std::vector<s25t::element> mles;
  std::vector<unsigned> product_terms;

  SECTION("we can map a single element mle") {
    mles = {0x123_s25};
    product_terms = {0};
    polynomial_mapper<1, T> m{
        .mles = mles.data(),
        .product_terms = product_terms.data(),
        .split = 1,
        .n = 1,
    };
    auto p = m.map_index(0);
    REQUIRE(p[0] == mles[0]);
    REQUIRE(p[1] == -mles[0]);
  }

  SECTION("we can map an mle with two elements") {
    mles = {0x123_s25, 0x456_s25};
    product_terms = {0};
    polynomial_mapper<1, T> m{
        .mles = mles.data(),
        .product_terms = product_terms.data(),
        .split = 1,
        .n = 2,
    };
    auto p = m.map_index(0);
    REQUIRE(p[0] == mles[0]);
    REQUIRE(p[1] == mles[1] - mles[0]);
  }
}
