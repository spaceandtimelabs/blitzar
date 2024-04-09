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
#include "sxt/curve_bng1_32/operation/double.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve_bng1_32/constant/generator.h"
#include "sxt/curve_bng1_32/property/curve.h"
#include "sxt/curve_bng1_32/property/identity.h"
#include "sxt/curve_bng1_32/type/element_p2.h"
#include "sxt/field32/constant/zero.h"

using namespace sxt;
using namespace sxt::cn3o;

TEST_CASE("doubling a projective element") {
  SECTION("preserves the identity") {
    cn3t::element_p2 identity_double;

    double_element(identity_double, cn3t::element_p2::identity());

    REQUIRE(cn3p::is_identity(identity_double));
    REQUIRE(cn3p::is_on_curve(identity_double));
  }

  SECTION("preserves the generator") {
    cn3t::element_p2 generator_double;

    double_element(generator_double, cn3cn::generator_p2_v);

    REQUIRE(!cn3p::is_identity(generator_double));
    REQUIRE(cn3p::is_on_curve(generator_double));
  }

  SECTION("produces double the generator") {
    constexpr cn3t::element_p2 expected{{0xc3e7ea38, 0xe10460b6, 0x438e5469, 0xbc0b548b, 0x0c0ac2ec,
                                         0xc2822db4, 0x098d014d, 0x13227397},
                                        {0xd87cfd47, 0x3c208c16, 0x6871ca8d, 0x97816a91, 0x8181585d,
                                         0xb85045b6, 0xe131a029, 0x04644e72},
                                        {0xc58f0d9d, 0xd35d438d, 0xf5c70b3d, 0x0a78eb28, 0x7879462c,
                                         0x666ea36f, 0x9a07df2f, 0x0e0a77c1}};
    cn3t::element_p2 generator_double;

    double_element(generator_double, cn3cn::generator_p2_v);

    REQUIRE(expected == generator_double);
  }

  SECTION("produces the identity when Z is the zero element") {
    constexpr cn3t::element_p2 p{cn3cn::generator_p2_v.X, cn3cn::generator_p2_v.Y, f32cn::zero_v};
    cn3t::element_p2 expect_identity;

    double_element(expect_identity, p);

    REQUIRE(cn3p::is_on_curve(p));
    REQUIRE(expect_identity == cn3t::element_p2::identity());
  }
}
