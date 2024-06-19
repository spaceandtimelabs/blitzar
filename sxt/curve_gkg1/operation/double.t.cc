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
#include "sxt/curve_gkg1/operation/double.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve_gkg1/constant/generator.h"
#include "sxt/curve_gkg1/property/curve.h"
#include "sxt/curve_gkg1/property/identity.h"
#include "sxt/curve_gkg1/type/element_p2.h"
#include "sxt/field25/constant/zero.h"

using namespace sxt;
using namespace sxt::ck1o;

TEST_CASE("doubling a projective element") {
  SECTION("preserves the identity") {
    ck1t::element_p2 identity_double;

    double_element(identity_double, ck1t::element_p2::identity());

    REQUIRE(ck1p::is_identity(identity_double));
    REQUIRE(ck1p::is_on_curve(identity_double));
  }

  SECTION("preserves the generator") {
    ck1t::element_p2 generator_double;

    double_element(generator_double, ck1cn::generator_p2_v);

    REQUIRE(!ck1p::is_identity(generator_double));
    REQUIRE(ck1p::is_on_curve(generator_double));
  }

  SECTION("produces double the generator") {
    constexpr ck1t::element_p2 expected{
        {0xe10460b6c3e7ea38, 0xbc0b548b438e5469, 0xc2822db40c0ac2ec, 0x13227397098d014d},
        {0x3c208c16d87cfd47, 0x97816a916871ca8d, 0xb85045b68181585d, 0x04644e72e131a029},
        {0xd35d438dc58f0d9d, 0x0a78eb28f5c70b3d, 0x666ea36f7879462c, 0xe0a77c19a07df2f}};
    ck1t::element_p2 generator_double;

    double_element(generator_double, ck1cn::generator_p2_v);

    REQUIRE(expected == generator_double);
  }

  SECTION("produces the identity when Z is the zero element") {
    constexpr ck1t::element_p2 p{ck1cn::generator_p2_v.X, ck1cn::generator_p2_v.Y, f25cn::zero_v};
    ck1t::element_p2 expect_identity;

    double_element(expect_identity, p);

    REQUIRE(ck1p::is_on_curve(p));
    REQUIRE(expect_identity == ck1t::element_p2::identity());
  }
}
