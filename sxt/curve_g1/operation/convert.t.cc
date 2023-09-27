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
/*
 * Adopted from zcash/librustzcash
 *
 * Copyright (c) 2017
 * Zcash Company
 *
 * See third_party/license/zcash.LICENSE
 */
#include "sxt/curve_g1/operation/convert.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve_g1/constant/generator.h"
#include "sxt/curve_g1/property/curve.h"
#include "sxt/curve_g1/property/identity.h"
#include "sxt/curve_g1/type/element_affine.h"
#include "sxt/curve_g1/type/element_p2.h"
#include "sxt/field12/operation/mul.h"
#include "sxt/field12/type/element.h"

using namespace sxt;
using namespace sxt::cg1o;

TEST_CASE("conversion from projective to affine elements") {
  cg1t::element_p2 generator_projective{cg1cn::generator_p2_v};
  cg1t::element_p2 identity_projective{cg1t::element_p2::identity()};

  SECTION("keeps the generator on the curve") {
    cg1t::element_affine generator_affine;

    convert(generator_affine, generator_projective);

    REQUIRE(cg1p::is_on_curve(generator_affine));
    REQUIRE(!cg1p::is_identity(generator_affine));
  }

  SECTION("keeps the identity on the curve") {
    cg1t::element_affine identity_affine;

    convert(identity_affine, identity_projective);

    REQUIRE(cg1p::is_on_curve(identity_affine));
    REQUIRE(cg1p::is_identity(identity_affine));
  }

  SECTION("can handle a projected generator coordinate") {
    constexpr f12t::element z{0xba7afa1f9a6fe250, 0xfa0f5b595eafe731, 0x3bdc477694c306e7,
                              0x2149be4b3949fa24, 0x64aa6e0649b2078c, 0x12b108ac33643c3e};

    f12t::element gpx_z;
    f12t::element gpy_z;
    f12o::mul(gpx_z, generator_projective.X, z);
    f12o::mul(gpy_z, generator_projective.Y, z);
    cg1t::element_p2 projective_pt{gpx_z, gpy_z, z};

    cg1t::element_affine affine_pt;

    convert(affine_pt, projective_pt);

    REQUIRE(cg1p::is_on_curve(affine_pt));
    REQUIRE(affine_pt == cg1cn::generator_affine_v);
  }
}
