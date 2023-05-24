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
#include "sxt/proof/inner_product/generator_fold.h"

#include <vector>

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/operation/overload.h"
#include "sxt/ristretto/operation/compression.h"
#include "sxt/ristretto/type/literal.h"
#include "sxt/scalar25/constant/max_bits.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/random/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfip;
using s25t::operator""_s25;
using rstt::operator""_rs;

TEST_CASE("we can compute the bitwise decomposition of a multiexponentiation fold") {
  unsigned data[s25cn::max_bits_v];
  basct::span<unsigned> decomposition{data};

  auto one = 0x1_s25;
  auto zero = 0x0_s25;

  SECTION("we handle the case of zero exponents") {
    decompose_generator_fold(decomposition, zero, zero);
    REQUIRE(decomposition.empty());
  }

  SECTION("we handle the case of single a single bit set") {
    decompose_generator_fold(decomposition, one, zero);
    REQUIRE(std::vector<unsigned>{decomposition.begin(), decomposition.end()} ==
            std::vector<unsigned>{1});

    decomposition = data;
    decompose_generator_fold(decomposition, zero, one);
    REQUIRE(std::vector<unsigned>{decomposition.begin(), decomposition.end()} ==
            std::vector<unsigned>{2});

    decomposition = data;
    decompose_generator_fold(decomposition, one, one);
    REQUIRE(std::vector<unsigned>{decomposition.begin(), decomposition.end()} ==
            std::vector<unsigned>{3});
  }

  SECTION("we handle the largest possible element") {
    decompose_generator_fold(decomposition, -one, zero);
    REQUIRE(decomposition.size() == s25cn::max_bits_v);
    REQUIRE(decomposition[s25cn::max_bits_v - 1] == 1);
  }
}

TEST_CASE("we can fold generator elements") {
  unsigned data[s25cn::max_bits_v];
  basct::span<unsigned> decomposition{data};

  auto g_low = 0x234_rs;
  auto g_high = 0x987_rs;
  auto res = 0x9999999999_rs;

  SECTION("we can fold elements with zero multipliers") {
    decomposition = {};
    fold_generators(res, decomposition, g_low, g_high);
  }

  SECTION("we handle large multipliers") {
    decompose_generator_fold(decomposition, -0x1_s25, -0x1_s25);
    fold_generators(res, decomposition, g_low, g_high);
    rstt::compressed_element expected, res_p;
    rsto::compress(expected, -g_low - g_high);
    rsto::compress(res_p, res);
    REQUIRE(res_p == expected);
  }

  SECTION("we handle random multipliers") {
    s25t::element low, high;
    basn::fast_random_number_generator rng{1234, 9879};
    s25rn::generate_random_element(low, rng);
    s25rn::generate_random_element(high, rng);
    decompose_generator_fold(decomposition, low, high);
    fold_generators(res, decomposition, g_low, g_high);
    rstt::compressed_element expected, res_p;
    rsto::compress(expected, low * g_low + high * g_high);
    rsto::compress(res_p, res);
    REQUIRE(res_p == expected);
  }
}
