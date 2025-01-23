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

#include <iostream>
#include <utility>
#include <vector>

#include "sxt/base/container/span_utility.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/proof/sumcheck/chunked_gpu_driver.h"
#include "sxt/proof/sumcheck/cpu_driver.h"
#include "sxt/proof/sumcheck/gpu_driver.h"
#include "sxt/proof/sumcheck/polynomial_utility.h"
#include "sxt/proof/sumcheck/sumcheck_random.h"
#include "sxt/proof/sumcheck/verification.h"
#include "sxt/proof/transcript/transcript.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfsk;
using s25t::operator""_s25;

static void test_proof(const driver& drv) noexcept {
  prft::transcript transcript{"abc"};
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

#if 0
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
#endif

  SECTION("we can verify random sumcheck problems") {
    basn::fast_random_number_generator rng{1, 2};

    for (unsigned i = 0; i < 10; ++i) {
      random_sumcheck_descriptor descriptor;
      unsigned n;
      generate_random_sumcheck_problem(mles, product_table, product_terms, n, rng, descriptor);
      std::println("num_mles = {}", mles.size() / n);

      unsigned polynomial_length = 0;
      for (auto [_, len] : product_table) {
        polynomial_length = std::max(polynomial_length, len + 1u);
      }

      auto num_variables = n == 1 ? 1 : basn::ceil_log2(n);
      evaluation_point.resize(num_variables);
      polynomials.resize(polynomial_length * num_variables);

      // prove
      {
        prft::transcript transcript{"abc"};
        auto fut = prove_sum(polynomials, evaluation_point, transcript, drv, mles, product_table,
                             product_terms, n);
        xens::get_scheduler().run();
      }

      // we can verify
      {
        prft::transcript transcript{"abc"};
        s25t::element expected_sum;
        sum_polynomial_01(expected_sum, basct::subspan(polynomials, 0, polynomial_length));
        std::cout << "expected sum: " << expected_sum << std::endl;
        auto valid = verify_sumcheck_no_evaluation(expected_sum, evaluation_point, transcript,
                                                   polynomials, polynomial_length - 1u);
        std::cerr << "v: " << valid << std::endl;
        REQUIRE(valid);
      }

      // verification fails if we break the proof
      {
        prft::transcript transcript{"abc"};
        s25t::element expected_sum;
        sum_polynomial_01(expected_sum, basct::subspan(polynomials, 0, polynomial_length));
        polynomials[polynomials.size() - 1] = polynomials[0] + polynomials[1];
        auto valid = verify_sumcheck_no_evaluation(expected_sum, evaluation_point, transcript,
                                                   polynomials, polynomial_length - 1u);
        REQUIRE(!valid);
      }
    }
  }
}

TEST_CASE("we can create a sumcheck proof") {
  SECTION("we can prove with the cpu driver") {
    cpu_driver drv;
    test_proof(drv);
  }

  SECTION("we can prove with the gpu driver") {
    gpu_driver drv;
    test_proof(drv);
  }

#if 0
  SECTION("we can prove with the chunked gpu driver") {
    chunked_gpu_driver drv{0.0};
    test_proof(drv);
  }
#endif
}
