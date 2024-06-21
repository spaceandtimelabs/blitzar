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
#include "sxt/curve_gkg1/property/curve.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve_gkg1/constant/b.h"
#include "sxt/curve_gkg1/constant/generator.h"
#include "sxt/curve_gkg1/type/element_affine.h"
#include "sxt/curve_gkg1/type/element_p2.h"
#include "sxt/fieldgk/constant/one.h"
#include "sxt/fieldgk/operation/mul.h"
#include "sxt/fieldgk/operation/neg.h"

using namespace sxt;
using namespace sxt::ck1p;

TEST_CASE("an affine element") {
  SECTION("equal to the generator is on the curve") {
    REQUIRE(is_on_curve(ck1cn::generator_affine_v));
  }

  SECTION("equal to the identity is on the curve") {
    REQUIRE(is_on_curve(ck1t::element_affine::identity()));
  }

  SECTION("equal to (1,1) is not on the curve") {
    constexpr ck1t::element_affine one_one{fgkcn::one_v, fgkcn::one_v, false};

    REQUIRE(!is_on_curve(one_one));
  }
}

TEST_CASE("a projective element") {
  SECTION("equal to the generator is on the curve") { REQUIRE(is_on_curve(ck1cn::generator_p2_v)); }

  SECTION("equal to the generator projected by z is on the curve") {
    // z is arbitrarily chosen to be 3 in Montgomery form for this section of the test.
    constexpr fgkt::element z{ck1cn::b_v};
    fgkt::element x_projected;
    fgkt::element y_projected;
    fgko::mul(x_projected, ck1cn::generator_p2_v.X, z);
    fgko::mul(y_projected, ck1cn::generator_p2_v.Y, z);
    ck1t::element_p2 generator_projected{x_projected, y_projected, z};

    REQUIRE(is_on_curve(generator_projected));

    fgkt::element neg_y;
    fgko::neg(neg_y, generator_projected.Y);
    generator_projected.Y = neg_y;

    REQUIRE(is_on_curve(generator_projected));

    fgkt::element neg_x;
    fgko::neg(neg_x, generator_projected.X);
    generator_projected.X = neg_x;

    REQUIRE(!is_on_curve(generator_projected));
  }

  SECTION("equal to the identity is on the curve") {
    REQUIRE(is_on_curve(ck1t::element_p2::identity()));
  }

  SECTION("equal to (1,1,1) is not on the curve") {
    constexpr ck1t::element_p2 one_one{fgkcn::one_v, fgkcn::one_v, fgkcn::one_v};

    REQUIRE(!is_on_curve(one_one));
  }
}
