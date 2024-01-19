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
/**
 * Adopted from zcash/librustzcash
 *
 * Copyright (c) 2017
 * Zcash Company
 *
 * See third_party/license/zcash.LICENSE
 */
#include "sxt/curve_g1/type/conversion_utility.h"

#include "sxt/base/container/span.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/curve_g1/type/element_affine.h"
#include "sxt/curve_g1/type/element_p2.h"
#include "sxt/field12/operation/mul.h"
#include "sxt/field12/type/element.h"

using namespace sxt;
using namespace sxt::cg1t;

constexpr f12t::element generator_x{0x5cb38790fd530c16, 0x7817fc679976fff5, 0x154f95c7143ba1c1,
                                    0xf0ae6acdf3d0e747, 0xedce6ecc21dbf440, 0x120177419e0bfb75};

constexpr f12t::element generator_y{0xbaac93d50ce72271, 0x8c22631a7918fd8e, 0xdd595f13570725ce,
                                    0x51ac582950405194, 0x0e1c8c3fad0059c0, 0x0bbc3efc5008a26a};

constexpr element_p2 generator_projective{generator_x, generator_y, f12cn::one_v};
constexpr element_p2 identity_projective{element_p2::identity()};

constexpr element_affine generator_affine{generator_x, generator_y, false};
constexpr element_affine identity_affine{element_affine::identity()};

TEST_CASE("conversion from projective to affine elements") {
  SECTION("does not change the generator") {
    element_affine result_affine;

    to_element_affine(result_affine, generator_projective);

    REQUIRE(result_affine == generator_affine);
  }

  SECTION("does not change the identity") {
    element_affine result_affine;

    to_element_affine(result_affine, identity_projective);

    REQUIRE(result_affine == identity_affine);
  }

  SECTION("does not change a projected generator coordinate") {
    constexpr f12t::element z{0xba7afa1f9a6fe250, 0xfa0f5b595eafe731, 0x3bdc477694c306e7,
                              0x2149be4b3949fa24, 0x64aa6e0649b2078c, 0x12b108ac33643c3e};

    f12t::element gpx_z;
    f12t::element gpy_z;
    f12o::mul(gpx_z, generator_projective.X, z);
    f12o::mul(gpy_z, generator_projective.Y, z);
    element_p2 projective_pt{gpx_z, gpy_z, z};

    element_affine affine_pt;

    to_element_affine(affine_pt, projective_pt);

    REQUIRE(affine_pt == generator_affine);
  }
}

TEST_CASE("conversion from affine to projective elements") {
  SECTION("does not change the generator") {
    element_p2 result_projective;

    to_element_p2(result_projective, generator_affine);

    REQUIRE(result_projective == generator_projective);
  }

  SECTION("does not change the identity") {
    element_p2 result_projective;

    to_element_p2(result_projective, identity_affine);

    REQUIRE(result_projective == identity_projective);
  }
}

TEST_CASE("batch conversion from affine to projective elements") {
  SECTION("does not change the generator") {
    std::vector<element_p2> res_vec{element_p2{}, element_p2{}};
    basct::span<element_p2> results{res_vec};
    const std::vector<element_affine> gen_vec{generator_affine, generator_affine};
    basct::cspan<element_affine> generators{gen_vec};

    batch_to_element_p2(results, generators);

    REQUIRE(results.size() == 2);
    REQUIRE(results[0] == generator_projective);
    REQUIRE(results[1] == generator_projective);
  }

  SECTION("does not change the identity") {
    std::vector<element_p2> res_vec{element_p2{}, element_p2{}};
    basct::span<element_p2> results{res_vec};
    const std::vector<element_affine> gen_vec{identity_affine, identity_affine};
    basct::cspan<element_affine> generators{gen_vec};

    batch_to_element_p2(results, generators);

    REQUIRE(results.size() == 2);
    REQUIRE(results[0] == identity_projective);
    REQUIRE(results[1] == identity_projective);
  }
}
