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
#include "cbindings/sumcheck.h"

#include <vector>

#include "cbindings/backend.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/proof/sumcheck/reference_transcript.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using s25t::operator""_s25;

TEST_CASE("we can create sumcheck proofs") {
  prft::transcript base_transcript{"abc"};
  prfsk::reference_transcript transcript{base_transcript};

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
  sumcheck_descriptor descriptor{
      .mles = mles.data(),
      .product_table = product_table.data(),
      .product_terms = product_terms.data(),
      .n = 2,
      .num_mles = 1,
      .num_products = 1,
      .num_product_terms = 1,
      .round_degree = 1,
  };

  auto f = [](s25t::element* r, void* context, const s25t::element* polynomial,
              unsigned polynomial_len) noexcept {
    static_cast<prfsk::reference_transcript*>(context)->round_challenge(
        *r, {polynomial, polynomial_len});
  };

  SECTION("we can prove a sum with n=2 on GPU") {
    cbn::reset_backend_for_testing();
    const sxt_config config = {SXT_GPU_BACKEND, 0};
    REQUIRE(sxt_init(&config) == 0);

    sxt_prove_sumcheck(polynomials.data(), evaluation_point.data(), SXT_FIELD_SCALAR255,
                       &descriptor, reinterpret_cast<void*>(+f), &transcript);
    REQUIRE(polynomials[0] == mles[0]);
    REQUIRE(polynomials[1] == mles[1] - mles[0]);
    {
      prft::transcript base_transcript_p{"abc"};
      prfsk::reference_transcript transcript_p{base_transcript_p};
      s25t::element r;
      transcript_p.round_challenge(r, polynomials);
      REQUIRE(evaluation_point[0] == r);
    }
  }

  SECTION("we can prove a sum with n=2 on CPU") {
    cbn::reset_backend_for_testing();
    const sxt_config config = {SXT_CPU_BACKEND, 0};
    REQUIRE(sxt_init(&config) == 0);

    sxt_prove_sumcheck(polynomials.data(), evaluation_point.data(), SXT_FIELD_SCALAR255,
                       &descriptor, reinterpret_cast<void*>(+f), &transcript);
    REQUIRE(polynomials[0] == mles[0]);
    REQUIRE(polynomials[1] == mles[1] - mles[0]);
    {
      prft::transcript base_transcript_p{"abc"};
      prfsk::reference_transcript transcript_p{base_transcript_p};
      s25t::element r;
      transcript_p.round_challenge(r, polynomials);
      REQUIRE(evaluation_point[0] == r);
    }
  }
}
