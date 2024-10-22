/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
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
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfsk;
using s25t::operator""_s25;

TEST_CASE("we perform basic operations on polynomials") {
  s25t::element e;

  std::vector<s25t::element> p;

  SECTION("we can compute the 0-1 sum of a zero polynomials") {
    sum_polynomial_01(e, p);
    REQUIRE(e == 0x0_s25);
  }

  SECTION("we can compute the 0-1 sum of a constant polynomial") {
    p = {0x123_s25};
    sum_polynomial_01(e, p);
    REQUIRE(e == 0x123_s25);
  }

  SECTION("we can compute the 0-1 sum of a 1 degree polynomial") {
    p = {0x123_s25, 0x456_s25};
    sum_polynomial_01(e, p);
    REQUIRE(e == 0x123_s25 + 0x456_s25);
  }

  SECTION("we can evaluate the zero polynomial") {
    evaluate_polynomial(e, p, 0x123_s25);
    REQUIRE(e == 0x0_s25);
  }

  SECTION("we can evaluate a constant polynomial") {
    p = {0x123_s25};
    evaluate_polynomial(e, p, 0x321_s25);
    REQUIRE(e == 0x123_s25);
  }

  SECTION("we can evaluate a polynomial of degree 1") {
    p = {0x123_s25, 0x456_s25};
    evaluate_polynomial(e, p, 0x321_s25);
    REQUIRE(e == 0x123_s25 + 0x456_s25 * 0x321_s25);
  }

  SECTION("we can evaluate a polynomial of degree 2") {
    p = {0x123_s25, 0x456_s25, 0x789_s25};
    evaluate_polynomial(e, p, 0x321_s25);
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
    expand_products(p, mles.data(), 2, 1, terms);
    REQUIRE(p[0] == mles[0]);
    REQUIRE(p[1] == mles[1] - mles[0]);
  }
}
