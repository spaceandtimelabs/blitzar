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
#include "sxt/curve_bng1_32/type/element_p2.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/field32/constant/one.h"
#include "sxt/field32/constant/zero.h"
#include "sxt/field32/operation/mul.h"
#include "sxt/field32/random/element.h"
#include "sxt/field32/type/element.h"

using namespace sxt;
using namespace sxt::cn3t;

TEST_CASE("projective element equality") {
  SECTION("can distinguish the generator from the identity") {
    constexpr element_p2 a{f32cn::one_v,
                           {0x8b1e1b3a, 0xa6ba871b, 0xeb8e167b, 0x14f1d651, 0xf0f28c58, 0xccdd46de,
                            0x340fbe5e, 0x1c14ef83},
                           f32cn::one_v};
    constexpr element_p2 b{element_p2::identity()};

    REQUIRE(a == a);
    REQUIRE(b == b);
    REQUIRE(a != b);
    REQUIRE(b != a);

    // Project the generator with a random field element.
    f32t::element z;
    basn::fast_random_number_generator rng{1, 2};
    f32rn::generate_random_element(z, rng);

    f32t::element ax_z;
    f32o::mul(ax_z, a.X, z);

    f32t::element ay_z;
    f32o::mul(ay_z, a.Y, z);

    element_p2 c{ax_z, ay_z, z};

    REQUIRE(a == c);
    REQUIRE(b != c);
    REQUIRE(c == a);
    REQUIRE(c != b);
  }
}
