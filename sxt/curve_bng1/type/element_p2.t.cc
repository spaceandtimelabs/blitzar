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
#include "sxt/curve_bng1/type/element_p2.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/field25/constant/one.h"
#include "sxt/field25/constant/zero.h"
#include "sxt/field25/operation/mul.h"
#include "sxt/field25/random/element.h"
#include "sxt/field25/type/element.h"
#include "sxt/field25/type/literal.h"

using namespace sxt;
using namespace sxt::cn1t;
using f25t::operator""_f25;

TEST_CASE("projective element equality") {
  SECTION("can distinguish the generator from the identity") {
    constexpr element_p2 a{
        f25cn::one_v,
        {0xa6ba871b8b1e1b3a, 0x14f1d651eb8e167b, 0xccdd46def0f28c58, 0x1c14ef83340fbe5e},
        f25cn::one_v};
    constexpr element_p2 b{element_p2::identity()};

    REQUIRE(a == a);
    REQUIRE(b == b);
    REQUIRE(a != b);
    REQUIRE(b != a);

    // Project the generator with a random field element.
    f25t::element z;
    basn::fast_random_number_generator rng{1, 2};
    f25rn::generate_random_element(z, rng);

    f25t::element ax_z;
    f25o::mul(ax_z, a.X, z);

    f25t::element ay_z;
    f25o::mul(ay_z, a.Y, z);

    element_p2 c{ax_z, ay_z, z};

    REQUIRE(a == c);
    REQUIRE(b != c);
    REQUIRE(c == a);
    REQUIRE(c != b);
  }
#if 0
0x30644e72e131a029b85045b63db22989a0ca93286ebbf9d9bc5fd495e1a92d47_f25
0x30644e72e131a029b85045b668f629f0d4ae64afaa9d239050df4dd8b672b147_f25
0x307084c8417aab1f9ec8866edfd3f27acc230000_f25
#endif
}
