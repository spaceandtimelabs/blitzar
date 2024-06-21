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
#include "sxt/curve_gkg1/operation/double.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve_gkg1/constant/generator.h"
#include "sxt/curve_gkg1/property/curve.h"
#include "sxt/curve_gkg1/property/identity.h"
#include "sxt/curve_gkg1/type/element_p2.h"
#include "sxt/fieldgk/constant/zero.h"
#include "sxt/fieldgk/type/literal.h"

using namespace sxt;
using namespace sxt::ck1o;
using fgkt::operator""_fgk;

TEST_CASE("doubling a projective element") {
  SECTION("preserves the identity") {
    ck1t::element_p2 identity_double;

    double_element(identity_double, ck1t::element_p2::identity());

    REQUIRE(ck1p::is_identity(identity_double));
    REQUIRE(ck1p::is_on_curve(identity_double));
  }

  SECTION("preserves the generator") {
    ck1t::element_p2 generator_double;

    double_element(generator_double, ck1cn::generator_p2_v);

    REQUIRE(!ck1p::is_identity(generator_double));
    REQUIRE(ck1p::is_on_curve(generator_double));
  }

  SECTION("produces double the generator") {
    ck1t::element_p2 expected{
        0x6ce1b0827aafa85ddeb49cdaa36306d19a74caa311e13d46d8bc688cdbffffe_fgk,
        0x1c122f81a3a14964909ede0ba2a6855fc93faf6fa1a788bf467be7e7a43f80ac_fgk, 0x1_fgk};
    ck1t::element_p2 generator_double;

    double_element(generator_double, ck1cn::generator_p2_v);

    REQUIRE(expected == generator_double);
  }

  SECTION("produces the identity when Z is the zero element") {
    constexpr ck1t::element_p2 p{ck1cn::generator_p2_v.X, ck1cn::generator_p2_v.Y, fgkcn::zero_v};
    ck1t::element_p2 expect_identity;

    double_element(expect_identity, p);

    REQUIRE(ck1p::is_on_curve(p));
    REQUIRE(expect_identity == ck1t::element_p2::identity());
  }
}
