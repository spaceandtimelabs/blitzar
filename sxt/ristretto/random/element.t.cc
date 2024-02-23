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
#include "sxt/ristretto/random/element.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/curve32/property/curve.h"
#include "sxt/curve32/type/element_p3.h"
#include "sxt/ristretto/type/compressed_element.h"

using namespace sxt;
using namespace sxt::rstrn;

TEST_CASE("we can generate random ristretto points") {
  basn::fast_random_number_generator rng{1, 2};

  SECTION("we can generate uncompressed elements") {
    c32t::element_p3 p1, p2;
    generate_random_element(p1, rng);
    generate_random_element(p2, rng);
    REQUIRE(p1 != p2);
    REQUIRE(c32p::is_on_curve(p1));
    REQUIRE(c32p::is_on_curve(p2));
  }

  SECTION("we can generate compressed elements") {
    rstt::compressed_element p1, p2;
    generate_random_element(p1, rng);
    generate_random_element(p2, rng);
    REQUIRE(p1 != p2);
  }

  SECTION("we can generate elements in bulk") {
    c32t::element_p3 px[2];
    generate_random_elements(px, rng);
    REQUIRE(px[0] != px[1]);
    REQUIRE(c32p::is_on_curve(px[0]));
    REQUIRE(c32p::is_on_curve(px[1]));
  }

  SECTION("we can generate compressed elements in bulk") {
    rstt::compressed_element px[2];
    generate_random_elements(px, rng);
    REQUIRE(px[0] != px[1]);
  }
}
