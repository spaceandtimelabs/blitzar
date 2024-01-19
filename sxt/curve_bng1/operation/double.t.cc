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
#include "sxt/curve_g1/operation/double.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve_g1/constant/generator.h"
#include "sxt/curve_g1/property/curve.h"
#include "sxt/curve_g1/property/identity.h"
#include "sxt/curve_g1/type/element_p2.h"
#include "sxt/field12/constant/zero.h"

using namespace sxt;
using namespace sxt::cg1o;

TEST_CASE("doubling a projective element") {
  SECTION("preserves the identity") {
    cg1t::element_p2 identity_double;

    double_element(identity_double, cg1t::element_p2::identity());

    REQUIRE(cg1p::is_identity(identity_double));
    REQUIRE(cg1p::is_on_curve(identity_double));
  }

  SECTION("preserves the generator") {
    cg1t::element_p2 generator_double;

    double_element(generator_double, cg1cn::generator_p2_v);

    REQUIRE(!cg1p::is_identity(generator_double));
    REQUIRE(cg1p::is_on_curve(generator_double));
  }

  SECTION("produces double the generator") {
    constexpr cg1t::element_p2 expected{
        {0xea99aa7b7fd2610f, 0x2d8f4ecb16a2c805, 0x6f5685b7bc2cce0c, 0xa00450614064a604,
         0x212102802cecd57b, 0x28576e1a7289e84},
        {0x111405902b1882bf, 0xa8cd3dd3a683ef06, 0x13639d9f8c73cebe, 0x3ac292fd4559e0f7,
         0x628c04a2e7fb8e20, 0x15c2f5f94df1f750},
        {0xfc9ac7edde06dbee, 0xfdc16d121f0f95e2, 0xa06d9a77977a906d, 0xef28f3348e385c64,
         0x8e0d17d2ffa3c835, 0x1700fc91a24772ec}};
    cg1t::element_p2 generator_double;

    double_element(generator_double, cg1cn::generator_p2_v);

    REQUIRE(expected == generator_double);
  }

  SECTION("produces the identity when Z is the zero element") {
    constexpr cg1t::element_p2 p{cg1cn::generator_p2_v.X, cg1cn::generator_p2_v.Y, f12cn::zero_v};
    cg1t::element_p2 expect_identity;

    double_element(expect_identity, p);

    REQUIRE(cg1p::is_on_curve(p));
    REQUIRE(expect_identity == cg1t::element_p2::identity());
  }
}
