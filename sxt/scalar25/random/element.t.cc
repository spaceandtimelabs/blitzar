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
#include "sxt/scalar25/random/element.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/type/element.h"

using namespace sxt;
using namespace sxt::s25rn;

TEST_CASE("we can generate random scalars") {
  basn::fast_random_number_generator rng{1, 2};

  SECTION("we don't generate the same scalar") {
    s25t::element x1, x2;
    generate_random_element(x1, rng);
    generate_random_element(x2, rng);
    REQUIRE(x1 != x2);
  }

  SECTION("we can generate random scalars in bulk") {
    s25t::element xx[2];
    generate_random_elements(xx, rng);
    REQUIRE(xx[0] != xx[1]);
  }
}
