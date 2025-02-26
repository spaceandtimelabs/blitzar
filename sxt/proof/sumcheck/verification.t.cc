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
#include "sxt/proof/sumcheck/verification.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/proof/sumcheck/reference_transcript.h"
#include "sxt/proof/transcript/transcript.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/realization/field.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfsk2;
using sxt::s25t::operator""_s25;

TEST_CASE("we can verify a sumcheck proof up to the polynomial evaluation") {
  using T = s25t::element;
  s25t::element expected_sum = 0x0_s25;
  std::vector<T> evaluation_point = {0x0_s25};
  prft::transcript base_transcript{"abc"};
  reference_transcript<T> transcript{base_transcript};
  std::vector<T> round_polynomials = {0x0_s25, 0x0_s25};

  SECTION("verification fails if dimensions don't match") {
    auto res = verify_sumcheck_no_evaluation<T>(expected_sum, evaluation_point, transcript,
                                                round_polynomials, 2);
    REQUIRE(!res);
  }

  SECTION("we can verify a single round") {
    auto res = verify_sumcheck_no_evaluation<T>(expected_sum, evaluation_point, transcript,
                                                round_polynomials, 1);
    REQUIRE(res);
    REQUIRE(evaluation_point[0] != 0x0_s25);
  }

  SECTION("verification fails if the round polynomial doesn't match the sum") {
    round_polynomials[1] = 0x1_s25;
    auto res = verify_sumcheck_no_evaluation<T>(expected_sum, evaluation_point, transcript,
                                                round_polynomials, 1);
    REQUIRE(!res);
  }

  SECTION("we can verify a sum with two rounds") {
    // Use the MLE:
    //    3(1-x1)(1-x2) + 5(1-x1)x2 -7x1(1-x2) -1x1x2
    round_polynomials.resize(4);

    // round 1
    round_polynomials[0] = 0x3_s25 + 0x5_s25;
    round_polynomials[1] = -0x3_s25 - 0x7_s25 - 0x5_s25 - 0x1_s25;

    // draw scalar
    s25t::element r;
    {
      prft::transcript base_transcript_p{"abc"};
      reference_transcript<T> transcript_p{base_transcript_p};
      transcript_p.init(2, 1);
      transcript_p.round_challenge(r, basct::span<T>{round_polynomials}.subspan(0, 2));
    }

    // round 2
    round_polynomials[2] = 0x3_s25 * (0x1_s25 - r) - 0x7_s25 * r;
    round_polynomials[3] =
        -0x3_s25 * (0x1_s25 - r) + 0x5_s25 * (0x1_s25 - r) + 0x7_s25 * r - 0x1_s25 * r;

    // prove
    evaluation_point.resize(2);
    auto res = verify_sumcheck_no_evaluation<T>(expected_sum, evaluation_point, transcript,
                                                round_polynomials, 1);
    REQUIRE(evaluation_point[0] == r);
    REQUIRE(res);
  }

  SECTION("sumcheck verification fails if the random scalar used is wrong") {
    // Use the MLE:
    //    3(1-x1)(1-x2) + 5(1-x1)x2 -7x1(1-x2) -1x1x2
    round_polynomials.resize(4);

    // round 1
    round_polynomials[0] = 0x3_s25 + 0x5_s25;
    round_polynomials[1] = -0x3_s25 - 0x7_s25 - 0x5_s25 - 0x1_s25;

    // draw scalar
    s25t::element r = 0x112233_s25;

    // round 2
    round_polynomials[2] = 0x3_s25 * (0x1_s25 - r) - 0x7_s25 * r;
    round_polynomials[3] =
        -0x3_s25 * (0x1_s25 - r) + 0x5_s25 * (0x1_s25 - r) + 0x7_s25 * r - 0x1_s25 * r;

    // prove
    evaluation_point.resize(2);
    auto res = verify_sumcheck_no_evaluation<T>(expected_sum, evaluation_point, transcript,
                                                round_polynomials, 1);
    REQUIRE(!res);
  }

  SECTION("we can verify a polynomial of degree 2 with one round") {
    // Use the MLEs:
    //    f(x1) = 3(1-x1) -7x1
    //    g(x1) = -2 (1 - x1) + 4 x1
    round_polynomials = {
        0x3_s25 * -0x2_s25,
        (-0x3_s25 - 0x7_s25) * -0x2_s25 + 0x3_s25 * (0x2_s25 + 0x4_s25),
        (-0x3_s25 - 0x7_s25) * (0x2_s25 + 0x4_s25),
    };
    expected_sum = 0x3_s25 * -0x2_s25 - 0x7_s25 * 0x4_s25;
    auto res = verify_sumcheck_no_evaluation<T>(expected_sum, evaluation_point, transcript,
                                                round_polynomials, 2);
    REQUIRE(res);
    REQUIRE(evaluation_point[0] != 0x0_s25);
  }
}
