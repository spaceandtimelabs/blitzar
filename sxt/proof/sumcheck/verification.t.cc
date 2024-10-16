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
#include "sxt/proof/sumcheck/verification.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/proof/transcript/transcript.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfsk;
using sxt::s25t::operator""_s25;

TEST_CASE("we can verify a sumcheck proof up to the polynomial evaluation") {
  s25t::element expected_sum = 0x0_s25;
  std::vector<s25t::element> evaluation_point = {0x0_s25};
  prft::transcript transcript{"abc"};
  std::vector<s25t::element> round_polynomials = {0x0_s25, 0x0_s25};

  SECTION("verification fails if dimensions don't match") {
    auto res = sxt::prfsk::verify_sumcheck_no_evaluation(expected_sum, evaluation_point, transcript,
                                                         round_polynomials, 2);
    REQUIRE(!res);
  }

  SECTION("we can verify a single round") {
    auto res = sxt::prfsk::verify_sumcheck_no_evaluation(expected_sum, evaluation_point, transcript,
                                                         round_polynomials, 1);
    REQUIRE(res);
    REQUIRE(evaluation_point[0] != 0x0_s25);
  }

  SECTION("verification fails if the round polynomial doesn't match the sum") {
    round_polynomials[1] = 0x1_s25;
    auto res = sxt::prfsk::verify_sumcheck_no_evaluation(expected_sum, evaluation_point, transcript,
                                                         round_polynomials, 1);
    REQUIRE(!res);
  }
}
