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
#include "sxt/curve_gk/operation/double.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve_gk/constant/generator.h"
#include "sxt/curve_gk/property/curve.h"
#include "sxt/curve_gk/property/identity.h"
#include "sxt/curve_gk/type/element_p2.h"
#include "sxt/fieldgk/constant/zero.h"
#include "sxt/fieldgk/type/literal.h"

using namespace sxt;
using namespace sxt::cgko;
using fgkt::operator""_fgk;

TEST_CASE("doubling a projective element") {
  SECTION("preserves the identity") {
    cgkt::element_p2 identity_double;

    double_element(identity_double, cgkt::element_p2::identity());

    REQUIRE(cgkp::is_identity(identity_double));
    REQUIRE(cgkp::is_on_curve(identity_double));
  }

  SECTION("preserves the generator") {
    cgkt::element_p2 generator_double;

    double_element(generator_double, cgkcn::generator_p2_v);

    REQUIRE(!cgkp::is_identity(generator_double));
    REQUIRE(cgkp::is_on_curve(generator_double));
  }

  SECTION("produces double the generator") {
    cgkt::element_p2 expected{
        0x6ce1b0827aafa85ddeb49cdaa36306d19a74caa311e13d46d8bc688cdbffffe_fgk,
        0x1c122f81a3a14964909ede0ba2a6855fc93faf6fa1a788bf467be7e7a43f80ac_fgk, 0x1_fgk};
    cgkt::element_p2 generator_double;

    double_element(generator_double, cgkcn::generator_p2_v);

    REQUIRE(expected == generator_double);
  }

  SECTION("produces the identity when Z is the zero element") {
    constexpr cgkt::element_p2 p{cgkcn::generator_p2_v.X, cgkcn::generator_p2_v.Y, fgkcn::zero_v};
    cgkt::element_p2 expect_identity;

    double_element(expect_identity, p);

    REQUIRE(cgkp::is_on_curve(p));
    REQUIRE(expect_identity == cgkt::element_p2::identity());
  }
}
