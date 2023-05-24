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
#include "sxt/proof/inner_product/fold.h"

#include <vector>

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/random/element.h"
#include "sxt/scalar25/type/element.h"

using namespace sxt;
using namespace sxt::prfip;

TEST_CASE("we can fold scalars") {
  basn::fast_random_number_generator rng{1, 2};
  s25t::element x, y;
  s25rn::generate_random_element(x, rng);
  s25rn::generate_random_element(y, rng);

  SECTION("we can fold 2 scalars") {
    std::vector<s25t::element> scalars(2);
    s25rn::generate_random_elements(scalars, rng);
    std::vector<s25t::element> scalars_p(2);
    basct::span<s25t::element> res = scalars_p;
    fold_scalars(res, scalars, x, y, 1);
    scalars_p.resize(res.size());

    std::vector<s25t::element> expected = {
        x * scalars[0] + y * scalars[1],
    };
    REQUIRE(scalars_p == expected);
  }

  SECTION("we can fold 3 scalars") {
    std::vector<s25t::element> scalars(3);
    s25rn::generate_random_elements(scalars, rng);
    std::vector<s25t::element> scalars_p(2);
    basct::span<s25t::element> res = scalars_p;
    fold_scalars(res, scalars, x, y, 2);
    scalars_p.resize(res.size());

    std::vector<s25t::element> expected = {
        x * scalars[0] + y * scalars[2],
        x * scalars[1],
    };
    REQUIRE(scalars_p == expected);
  }
}
