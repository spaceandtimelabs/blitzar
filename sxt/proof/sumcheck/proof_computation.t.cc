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
#include "sxt/proof/sumcheck/proof_computation.h"

#include <utility>
#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/proof/sumcheck/cpu_driver.h"
#include "sxt/proof/sumcheck/gpu_driver.h"
#include "sxt/proof/transcript/transcript.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfsk;
using s25t::operator""_s25;

TEST_CASE("we can create a sumcheck proof") {
  prft::transcript transcript{"abc"};
  /* cpu_driver drv; */
  gpu_driver drv;
  std::vector<s25t::element> polynomials(2);
  std::vector<s25t::element> evaluation_point(1);
  std::vector<s25t::element> mles = {
      0x8_s25,
      0x3_s25,
  };
  std::vector<std::pair<s25t::element, unsigned>> product_table = {
      {0x1_s25, 1},
  };
  std::vector<unsigned> product_terms = {0};

  SECTION("we can prove a sum with n=1") {
    auto fut = prove_sum(polynomials, evaluation_point, transcript, drv, mles, product_table,
                         product_terms, 1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(polynomials[0] == mles[0]);
    REQUIRE(polynomials[1] == -mles[0]);
  }

  SECTION("we can prove a sum with a single variable") {
    auto fut = prove_sum(polynomials, evaluation_point, transcript, drv, mles, product_table,
                         product_terms, 2);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(polynomials[0] == mles[0]);
    REQUIRE(polynomials[1] == mles[1] - mles[0]);
  }

  SECTION("we can prove a sum degree greater than 1") {
    product_table = {
        {0x1_s25, 2},
    };
    product_terms = {0, 0};
    polynomials.resize(3);
    auto fut = prove_sum(polynomials, evaluation_point, transcript, drv, mles, product_table,
                         product_terms, 2);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(polynomials[0] == mles[0] * mles[0]);
    REQUIRE(polynomials[1] == 0x2_s25 * (mles[1] - mles[0]) * mles[0]);
    REQUIRE(polynomials[2] == (mles[1] - mles[0]) * (mles[1] - mles[0]));
  }

  SECTION("we can prove a sum with multiple MLEs") {
    product_table = {
        {0x1_s25, 2},
    };
    product_terms = {0, 1};
    polynomials.resize(3);
    mles.push_back(0x7_s25);
    mles.push_back(0x10_s25);
    auto fut = prove_sum(polynomials, evaluation_point, transcript, drv, mles, product_table,
                         product_terms, 2);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(polynomials[0] == mles[0] * mles[2]);
    REQUIRE(polynomials[1] == (mles[1] - mles[0]) * mles[2] + (mles[3] - mles[2]) * mles[0]);
    REQUIRE(polynomials[2] == (mles[1] - mles[0]) * (mles[3] - mles[2]));
  }

  SECTION("we can prove a sum where the term multiplier is different from one") {
    product_table[0].first = 0x2_s25;
    auto fut = prove_sum(polynomials, evaluation_point, transcript, drv, mles, product_table,
                         product_terms, 2);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(polynomials[0] == 0x2_s25 * mles[0]);
    REQUIRE(polynomials[1] == 0x2_s25 * (mles[1] - mles[0]));
  }

  SECTION("we can prove a sum with two variables") {
    mles.push_back(0x4_s25);
    mles.push_back(0x7_s25);
    polynomials.resize(4);
    evaluation_point.resize(2);
    auto fut = prove_sum(polynomials, evaluation_point, transcript, drv, mles, product_table,
                         product_terms, 4);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(polynomials[0] == mles[0] + mles[1]);
    REQUIRE(polynomials[1] == (mles[2] - mles[0]) + (mles[3] - mles[1]));

    auto r = evaluation_point[0];
    mles[0] = mles[0] * (0x1_s25 - r) + mles[2] * r;
    mles[1] = mles[1] * (0x1_s25 - r) + mles[3] * r;

    REQUIRE(polynomials[2] == mles[0]);
    REQUIRE(polynomials[3] == mles[1] - mles[0]);
  }

  SECTION("we can prove a sum with n=3") {
    mles.push_back(0x4_s25);
    polynomials.resize(4);
    evaluation_point.resize(2);
    auto fut = prove_sum(polynomials, evaluation_point, transcript, drv, mles, product_table,
                         product_terms, 3);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    REQUIRE(polynomials[0] == mles[0] + mles[1]);
    REQUIRE(polynomials[1] == (mles[2] - mles[0]) - mles[1]);

    auto r = evaluation_point[0];
    mles[0] = mles[0] * (0x1_s25 - r) + mles[2] * r;
    mles[1] = mles[1] * (0x1_s25 - r);

    REQUIRE(polynomials[2] == mles[0]);
    REQUIRE(polynomials[3] == mles[1] - mles[0]);
  }
}
