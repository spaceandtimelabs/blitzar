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
#include "sxt/proof/inner_product/verification_computation.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/operation/inv.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfip;
using sxt::s25t::operator""_s25;

TEST_CASE("we can compute the exponents for the final inner product commitment verification") {
  std::vector<s25t::element> exponents, x_vector, x_inv_vector, b_vector;
  auto ap_value = 0x100_s25;

  SECTION("for proofs of length 1, the exponents consist only of the inner product and the "
          "commitment of the original a vector") {
    exponents.resize(2);
    b_vector = {0x3_s25};
    compute_verification_exponents(exponents, x_vector, ap_value, b_vector);
    std::vector<s25t::element> expected = {
        ap_value * b_vector[0],
        ap_value,
    };
    REQUIRE(exponents == expected);
  }

  SECTION("we handle a single fold") {
    exponents.resize(5);
    b_vector = {0x3_s25, 0x897234_s25};
    x_vector = {0x97234_s25};
    x_inv_vector.resize(x_vector.size());
    s25o::batch_inv(x_inv_vector, x_vector);
    compute_verification_exponents(exponents, x_vector, ap_value, b_vector);

    auto folded_b = x_inv_vector[0] * b_vector[0] + x_vector[0] * b_vector[1];
    std::vector<s25t::element> expected = {
        ap_value * folded_b,                // Q
        ap_value * x_inv_vector[0],         // g0
        ap_value * x_vector[0],             // g1
        -x_vector[0] * x_vector[0],         // L0
        -x_inv_vector[0] * x_inv_vector[0], // R0
    };
    REQUIRE(exponents == expected);
  }

  SECTION("we handle more than one fold") {
    exponents.resize(9);
    b_vector = {0x3_s25, 0x897234_s25, 0x2376_s25};
    x_vector = {0x97234_s25, 0x58763_s25};
    x_inv_vector.resize(x_vector.size());
    s25o::batch_inv(x_inv_vector, x_vector);
    compute_verification_exponents(exponents, x_vector, ap_value, b_vector);

    auto folded_b = x_inv_vector[0] * x_inv_vector[1] * b_vector[0] +
                    x_inv_vector[0] * x_vector[1] * b_vector[1] +
                    x_vector[0] * x_inv_vector[1] * b_vector[2];
    std::vector<s25t::element> expected = {
        ap_value * folded_b,                          // Q
        ap_value * x_inv_vector[0] * x_inv_vector[1], // g0
        ap_value * x_inv_vector[0] * x_vector[1],     // g1
        ap_value * x_vector[0] * x_inv_vector[1],     // g2
        ap_value * x_vector[0] * x_vector[1],         // g3
        -x_vector[0] * x_vector[0],                   // L0
        -x_vector[1] * x_vector[1],                   // L1
        -x_inv_vector[0] * x_inv_vector[0],           // R0
        -x_inv_vector[1] * x_inv_vector[1],           // R1
    };
    REQUIRE(exponents == expected);
  }
}
