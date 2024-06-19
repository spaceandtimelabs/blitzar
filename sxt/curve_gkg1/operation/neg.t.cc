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
#include "sxt/curve_gkg1/operation/neg.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve_gkg1/constant/generator.h"
#include "sxt/curve_gkg1/operation/add.h"
#include "sxt/curve_gkg1/type/element_p2.h"

using namespace sxt;
using namespace sxt::ck1o;

TEST_CASE("negation on projective elements") {
  SECTION("produces the identity when summing the generator with its negation") {
    ck1t::element_p2 gen_neg;
    neg(gen_neg, ck1cn::generator_p2_v);

    ck1t::element_p2 expect_identity;
    add(expect_identity, ck1cn::generator_p2_v, gen_neg);

    REQUIRE(expect_identity == ck1t::element_p2::identity());
  }

  SECTION("can be done inplace") {
    ck1t::element_p2 ng;
    neg(ng, ck1cn::generator_p2_v);
    ck1t::element_p2 g{ck1cn::generator_p2_v};
    cneg(g, 1);

    REQUIRE(g == ng);

    g = ck1cn::generator_p2_v;
    cneg(g, 0);

    REQUIRE(g == ck1cn::generator_p2_v);
  }
}
