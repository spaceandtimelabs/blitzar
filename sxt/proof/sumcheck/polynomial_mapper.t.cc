/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
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

#include "sxt/algorithm/base/mapper.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfsk;
using s25t::operator""_s25;

TEST_CASE("we can map an index to expanded MLE products") {
  REQUIRE(algb::mapper<polynomial_mapper<2>>);

  std::vector<s25t::element> mles = {0x123_s25, 0x456_s25};
  std::vector<std::pair<s25t::element, unsigned>> product_table = {
      {0x1_s25, 1},
  };
  std::vector<unsigned> product_terms = {0};

  SECTION("we can map an index to an expanded polynomial") {
    polynomial_mapper<1> mapper{
        .mles = mles.data(),
        .product_table = product_table.data(),
        .product_terms = product_terms.data(),
        .num_products = 1,
        .mid = 1,
        .n = 2,
    };
    auto p = mapper.map_index(0);
    REQUIRE(p[0] == mles[0]);
    REQUIRE(p[1] == mles[1] - mles[0]);
  }
}
