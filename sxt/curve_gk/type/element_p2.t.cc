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
 * Adopted from zkcrypto/bls12_381
 *
 * Copyright (c) 2021
 * Sean Bowe <ewillbefull@gmail.com>
 * Jack Grigg <thestr4d@gmail.com>
 *
 * See third_party/license/zkcrypto.LICENSE
 */
#include "sxt/curve_gk/type/element_p2.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/fieldgk/constant/one.h"
#include "sxt/fieldgk/constant/zero.h"
#include "sxt/fieldgk/operation/mul.h"
#include "sxt/fieldgk/random/element.h"
#include "sxt/fieldgk/type/element.h"
#include "sxt/fieldgk/type/literal.h"

using namespace sxt;
using namespace sxt::cgkt;
using fgkt::operator""_fgk;

TEST_CASE("projective element equality") {
  SECTION("can distinguish the generator from the identity") {
    element_p2 generator{fgkcn::one_v,
                         0x14b34cf69dc25d68aa7b8cf435dfafbb23d3446f21c77dc311b2dff1448c41d8_fgk,
                         fgkcn::one_v};

    REQUIRE(generator == generator);
    REQUIRE(element_p2::identity() == element_p2::identity());
    REQUIRE(generator != element_p2::identity());
    REQUIRE(element_p2::identity() != generator);

    // Project the generator with a random field element.
    fgkt::element z;
    basn::fast_random_number_generator rng{1, 2};
    fgkrn::generate_random_element(z, rng);

    fgkt::element projx_z;
    fgko::mul(projx_z, generator.X, z);

    fgkt::element projy_z;
    fgko::mul(projy_z, generator.Y, z);

    element_p2 generator_projected{projx_z, projy_z, z};

    REQUIRE(generator == generator_projected);
    REQUIRE(element_p2::identity() != generator_projected);
    REQUIRE(generator_projected == generator);
    REQUIRE(generator_projected != element_p2::identity());
  }
}

TEST_CASE("we can convert between elements") {
  SECTION("we can convert the identity element") {
    auto id = element_p2::identity();
    auto id_p = element_p2{static_cast<compact_element>(id)};
    REQUIRE(id == id_p);
  }

  SECTION("we can covert an arbitrary element") {
    element_p2 e{0x21c12e04bad4c2156ce3b4a8a3308c4fb4aacdadef62a450b495e0f6a86535ac_fgk,
                 0x150b08bd8caf3a73f977c855a4550b9f0d2599a2a2c024609bbc45ffb56f854a_fgk,
                 0x87110725e6d5daae7f0df824436892b177b7196fd22d9c2d59d6561cf0e2ada_fgk};
    auto ep = element_p2{static_cast<compact_element>(e)};
    REQUIRE(e == ep);
  }
}
