/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2025-present Space and Time Labs, Inc.
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
#include "sxt/proof/sumcheck/polynomial_utility.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/realization/field.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfsk;
using s25t::operator""_s25;

using T = s25t::element;

TEST_CASE("we perform basic operations on polynomials") {

  s25t::element e;

  std::vector<s25t::element> p;

  SECTION("we can compute the 0-1 sum of a zero polynomials") {
    sum_polynomial_01<T>(e, p);
    REQUIRE(e == 0x0_s25);
  }

  SECTION("we can compute the 0-1 sum of a constant polynomial") {
    p = {0x123_s25};
    sum_polynomial_01<T>(e, p);
    REQUIRE(e == 0x246_s25);
  }

  SECTION("we can compute the 0-1 sum of a 1 degree polynomial") {
    p = {0x123_s25, 0x456_s25};
    sum_polynomial_01<T>(e, p);
    REQUIRE(e == 0x246_s25 + 0x456_s25);
  }

  SECTION("we can evaluate the zero polynomial") {
    evaluate_polynomial<T>(e, p, 0x123_s25);
    REQUIRE(e == 0x0_s25);
  }

  SECTION("we can evaluate a constant polynomial") {
    p = {0x123_s25};
    evaluate_polynomial<T>(e, p, 0x321_s25);
    REQUIRE(e == 0x123_s25);
  }

  SECTION("we can evaluate a polynomial of degree 1") {
    p = {0x123_s25, 0x456_s25};
    evaluate_polynomial<T>(e, p, 0x321_s25);
    REQUIRE(e == 0x123_s25 + 0x456_s25 * 0x321_s25);
  }

  SECTION("we can evaluate a polynomial of degree 2") {
    p = {0x123_s25, 0x456_s25, 0x789_s25};
    evaluate_polynomial<T>(e, p, 0x321_s25);
    REQUIRE(e == 0x123_s25 + 0x456_s25 * 0x321_s25 + 0x789_s25 * 0x321_s25 * 0x321_s25);
  }
}

TEST_CASE("we can expand a product of MLEs") {
  std::vector<s25t::element> p;
  std::vector<s25t::element> mles;
  std::vector<unsigned> terms;

  SECTION("we can expand a single MLE") {
    p.resize(2);
    mles = {0x123_s25, 0x456_s25};
    terms = {0};
    expand_products<T>(p, mles.data(), 2, 1, terms);
    REQUIRE(p[0] == mles[0]);
    REQUIRE(p[1] == mles[1] - mles[0]);
  }

  SECTION("we can partially expand MLEs (where some terms are assumed to be zero)") {
    mles = {0x123_s25, 0x0_s25};
    p.resize(2);
    terms = {0};
    partial_expand_products<T>(p, mles.data(), 1, terms);

    std::vector<s25t::element> expected(2);
    expand_products<T>(expected, mles.data(), 2, 1, terms);
    REQUIRE(p == expected);
  }

  SECTION("we can expand two MLEs") {
    p.resize(3);
    mles = {0x123_s25, 0x456_s25, 0x1122_s25, 0x4455_s25};
    terms = {0, 1};
    expand_products<T>(p, mles.data(), 2, 1, terms);
    auto a1 = mles[0];
    auto a2 = mles[1] - mles[0];
    auto b1 = mles[2];
    auto b2 = mles[3] - mles[2];
    REQUIRE(p[0] == a1 * b1);
    REQUIRE(p[1] == a1 * b2 + a2 * b1);
    REQUIRE(p[2] == a2 * b2);
  }

  SECTION("we can expand three MLEs") {
    p.resize(4);
    mles = {0x123_s25, 0x456_s25, 0x1122_s25, 0x4455_s25, 0x2233_s25, 0x5566_s25};
    terms = {0, 1, 2};
    expand_products<T>(p, mles.data(), 2, 1, terms);
    auto a1 = mles[0];
    auto a2 = mles[1] - mles[0];
    auto b1 = mles[2];
    auto b2 = mles[3] - mles[2];
    auto c1 = mles[4];
    auto c2 = mles[5] - mles[4];
    REQUIRE(p[0] == a1 * b1 * c1);
    REQUIRE(p[1] == a1 * b1 * c2 + a1 * b2 * c1 + a2 * b1 * c1);
    REQUIRE(p[2] == a1 * b2 * c2 + a2 * b1 * c2 + a2 * b2 * c1);
    REQUIRE(p[3] == a2 * b2 * c2);
  }
}
