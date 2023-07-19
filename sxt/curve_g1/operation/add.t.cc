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
#include "sxt/curve_g1/operation/add.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve_g1/constant/generator.h"
#include "sxt/curve_g1/constant/identity.h"
#include "sxt/curve_g1/operation/double.h"
#include "sxt/curve_g1/property/curve.h"
#include "sxt/curve_g1/property/identity.h"
#include "sxt/curve_g1/type/element_p2.h"
#include "sxt/field12/operation/mul.h"
#include "sxt/field12/type/element.h"

using namespace sxt;
using namespace sxt::cg1o;

TEST_CASE("addition with projective elements") {
  SECTION("keeps the identity on the curve") {
    cg1t::element_p2 ret;
    add(ret, cg1cn::identity_p2_v, cg1cn::identity_p2_v);

    REQUIRE(cg1p::is_identity(ret));
    REQUIRE(cg1p::is_on_curve(ret));
  }

  SECTION("is commutative") {
    constexpr f12t::element z{0xba7afa1f9a6fe250, 0xfa0f5b595eafe731, 0x3bdc477694c306e7,
                              0x2149be4b3949fa24, 0x64aa6e0649b2078c, 0x12b108ac33643c3e};
    f12t::element x;
    f12t::element y;
    f12o::mul(x, cg1cn::generator_p2_v.X, z);
    f12o::mul(y, cg1cn::generator_p2_v.Y, z);
    const cg1t::element_p2 projected_generator{x, y, z};
    cg1t::element_p2 ret;

    add(ret, cg1cn::identity_p2_v, projected_generator);

    REQUIRE(!cg1p::is_identity(ret));
    REQUIRE(cg1p::is_on_curve(ret));
    REQUIRE(cg1cn::generator_p2_v == ret);

    // Switch summands
    add(ret, projected_generator, cg1cn::identity_p2_v);

    REQUIRE(!cg1p::is_identity(ret));
    REQUIRE(cg1p::is_on_curve(ret));
    REQUIRE(cg1cn::generator_p2_v == ret);
  }

  SECTION("can reproduce doubling results") {
    cg1t::element_p2 a;
    cg1t::element_p2 b;
    cg1t::element_p2 c;

    double_element(a, cg1cn::generator_p2_v); // a = 2g
    double_element(a, a);                     // a = 4g
    double_element(b, cg1cn::generator_p2_v); // b = 2g
    add(c, a, b);                             // c = 6g

    cg1t::element_p2 d{cg1cn::generator_p2_v};
    for (size_t i = 1; i < 6; ++i) {
      add(d, d, cg1cn::generator_p2_v);
    }

    REQUIRE(!cg1p::is_identity(c));
    REQUIRE(cg1p::is_on_curve(c));
    REQUIRE(!cg1p::is_identity(d));
    REQUIRE(cg1p::is_on_curve(d));
    REQUIRE(c == d);
  }
}

TEST_CASE("addition with mixed elements") {
  SECTION("keeps the identity on the curve") {
    cg1t::element_p2 ret;
    add(ret, cg1cn::identity_p2_v, cg1cn::identity_affine_v);

    REQUIRE(cg1p::is_identity(ret));
    REQUIRE(cg1p::is_on_curve(ret));
  }

  SECTION("keeps the generator on the curve") {
    constexpr f12t::element z{0xba7afa1f9a6fe250, 0xfa0f5b595eafe731, 0x3bdc477694c306e7,
                              0x2149be4b3949fa24, 0x64aa6e0649b2078c, 0x12b108ac33643c3e};
    f12t::element x;
    f12t::element y;
    f12o::mul(x, cg1cn::generator_p2_v.X, z);
    f12o::mul(y, cg1cn::generator_p2_v.Y, z);
    const cg1t::element_p2 projected_generator{x, y, z};
    cg1t::element_p2 ret;

    add(ret, projected_generator, cg1cn::identity_affine_v);

    REQUIRE(!cg1p::is_identity(ret));
    REQUIRE(cg1p::is_on_curve(ret));
    REQUIRE(cg1cn::generator_p2_v == ret);
  }

  SECTION("can reproduce doubling results") {
    cg1t::element_p2 a;
    cg1t::element_p2 b;
    cg1t::element_p2 c;

    double_element(a, cg1cn::generator_p2_v); // a = 2g
    double_element(a, a);                     // a = 4g
    double_element(b, cg1cn::generator_p2_v); // b = 2g
    add(c, a, b);                             // c = 6g

    cg1t::element_p2 d{cg1cn::generator_p2_v};
    for (size_t i = 1; i < 6; ++i) {
      add(d, d, cg1cn::generator_affine_v);
    }

    REQUIRE(!cg1p::is_identity(d));
    REQUIRE(cg1p::is_on_curve(d));
    REQUIRE(c == d);
  }
}
