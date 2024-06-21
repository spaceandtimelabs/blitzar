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
#include "sxt/curve_gkg1/type/conversion_utility.h"

#include "sxt/base/container/span.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/curve_gkg1/type/element_affine.h"
#include "sxt/curve_gkg1/type/element_p2.h"
#include "sxt/fieldgk/constant/one.h"
#include "sxt/fieldgk/operation/mul.h"
#include "sxt/fieldgk/random/element.h"
#include "sxt/fieldgk/type/element.h"

using namespace sxt;
using namespace sxt::ck1t;

constexpr fgkt::element generator_x{fgkcn::one_v};
constexpr fgkt::element generator_y{0xa6ba871b8b1e1b3a, 0x14f1d651eb8e167b, 0xccdd46def0f28c58,
                                    0x1c14ef83340fbe5e};

constexpr element_p2 generator_projective{generator_x, generator_y, fgkcn::one_v};
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
    fgkt::element z;
    basn::fast_random_number_generator rng{1, 2};
    fgkrn::generate_random_element(z, rng);

    fgkt::element gpx_z;
    fgkt::element gpy_z;
    fgko::mul(gpx_z, generator_projective.X, z);
    fgko::mul(gpy_z, generator_projective.Y, z);
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

TEST_CASE("batch conversion from projective to affine elements") {
  SECTION("does not change the generator") {
    std::vector<element_affine> res_vec{element_affine{}, element_affine{}};
    basct::span<element_affine> results{res_vec};
    const std::vector<element_p2> gen_vec{generator_projective, generator_projective};
    basct::cspan<element_p2> generators{gen_vec};

    batch_to_element_affine(results, generators);

    REQUIRE(results.size() == 2);
    REQUIRE(results[0] == generator_affine);
    REQUIRE(results[1] == generator_affine);
  }

  SECTION("does not change the identity") {
    std::vector<element_affine> res_vec{element_affine{}, element_affine{}};
    basct::span<element_affine> results{res_vec};
    const std::vector<element_p2> gen_vec{identity_projective, identity_projective};
    basct::cspan<element_p2> generators{gen_vec};

    batch_to_element_affine(results, generators);

    REQUIRE(results.size() == 2);
    REQUIRE(results[0] == identity_affine);
    REQUIRE(results[1] == identity_affine);
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
