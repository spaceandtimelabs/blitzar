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
#include "sxt/curve_bng1/operation/add.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/curve_bng1/constant/generator.h"
#include "sxt/curve_bng1/operation/double.h"
#include "sxt/curve_bng1/property/curve.h"
#include "sxt/curve_bng1/property/identity.h"
#include "sxt/curve_bng1/type/element_affine.h"
#include "sxt/curve_bng1/type/element_p2.h"
#include "sxt/field25/operation/mul.h"
#include "sxt/field25/random/element.h"
#include "sxt/field25/type/element.h"

using namespace sxt;
using namespace sxt::cn1o;

TEST_CASE("addition with projective elements") {
  SECTION("keeps the identity on the curve") {
    cn1t::element_p2 ret;
    add(ret, cn1t::element_p2::identity(), cn1t::element_p2::identity());

    REQUIRE(cn1p::is_identity(ret));
    REQUIRE(cn1p::is_on_curve(ret));
  }

  SECTION("is commutative") {
    f25t::element z;
    basn::fast_random_number_generator rng{1, 2};
    f25rn::generate_random_element(z, rng);

    f25t::element x;
    f25t::element y;
    f25o::mul(x, cn1cn::generator_p2_v.X, z);
    f25o::mul(y, cn1cn::generator_p2_v.Y, z);
    const cn1t::element_p2 projected_generator{x, y, z};
    cn1t::element_p2 ret;

    add(ret, cn1t::element_p2::identity(), projected_generator);

    REQUIRE(!cn1p::is_identity(ret));
    REQUIRE(cn1p::is_on_curve(ret));
    REQUIRE(cn1cn::generator_p2_v == ret);

    // Switch summands
    add(ret, projected_generator, cn1t::element_p2::identity());

    REQUIRE(!cn1p::is_identity(ret));
    REQUIRE(cn1p::is_on_curve(ret));
    REQUIRE(cn1cn::generator_p2_v == ret);
  }

  SECTION("can reproduce doubling results") {
    cn1t::element_p2 a;
    cn1t::element_p2 b;
    cn1t::element_p2 c;

    double_element(a, cn1cn::generator_p2_v); // a = 2g
    double_element(a, a);                     // a = 4g
    double_element(b, cn1cn::generator_p2_v); // b = 2g
    add(c, a, b);                             // c = 6g

    cn1t::element_p2 d{cn1cn::generator_p2_v};
    for (size_t i = 1; i < 6; ++i) {
      add(d, d, cn1cn::generator_p2_v);
    }

    REQUIRE(!cn1p::is_identity(c));
    REQUIRE(cn1p::is_on_curve(c));
    REQUIRE(!cn1p::is_identity(d));
    REQUIRE(cn1p::is_on_curve(d));
    REQUIRE(c == d);
  }

  SECTION("can be done inplace") {
    cn1t::element_p2 lhs{cn1t::element_p2::identity()};
    cn1t::element_p2 rhs{cn1cn::generator_p2_v};

    add_inplace(lhs, rhs);

    REQUIRE(lhs == cn1cn::generator_p2_v);
  }
}

TEST_CASE("addition with mixed elements") {
  SECTION("keeps the identity on the curve") {
    cn1t::element_p2 ret;
    add(ret, cn1t::element_p2::identity(), cn1t::element_affine::identity());

    REQUIRE(cn1p::is_identity(ret));
    REQUIRE(cn1p::is_on_curve(ret));
  }

  SECTION("keeps the generator on the curve") {
    f25t::element z;
    basn::fast_random_number_generator rng{1, 2};
    f25rn::generate_random_element(z, rng);

    f25t::element x;
    f25t::element y;
    f25o::mul(x, cn1cn::generator_p2_v.X, z);
    f25o::mul(y, cn1cn::generator_p2_v.Y, z);
    const cn1t::element_p2 projected_generator{x, y, z};
    cn1t::element_p2 ret;

    add(ret, projected_generator, cn1t::element_affine::identity());

    REQUIRE(!cn1p::is_identity(ret));
    REQUIRE(cn1p::is_on_curve(ret));
    REQUIRE(cn1cn::generator_p2_v == ret);
  }

  SECTION("can reproduce doubling results") {
    cn1t::element_p2 a;
    cn1t::element_p2 b;
    cn1t::element_p2 c;

    double_element(a, cn1cn::generator_p2_v); // a = 2g
    double_element(a, a);                     // a = 4g
    double_element(b, cn1cn::generator_p2_v); // b = 2g
    add(c, a, b);                             // c = 6g

    cn1t::element_p2 d{cn1cn::generator_p2_v};
    for (size_t i = 1; i < 6; ++i) {
      add(d, d, cn1cn::generator_affine_v);
    }

    REQUIRE(!cn1p::is_identity(d));
    REQUIRE(cn1p::is_on_curve(d));
    REQUIRE(c == d);
  }
}
