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
#include "sxt/fieldgk/operation/mul.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/fieldgk/base/constants.h"
#include "sxt/fieldgk/base/montgomery.h"
#include "sxt/fieldgk/constant/zero.h"
#include "sxt/fieldgk/random/element.h"
#include "sxt/fieldgk/type/element.h"
#include "sxt/fieldgk/type/literal.h"

using namespace sxt;
using namespace sxt::fgko;
using namespace sxt::fgkt;

TEST_CASE("multiplication") {
  SECTION("of a random field element and zero returns zero") {
    element a;
    basn::fast_random_number_generator rng{1, 2};
    fgkrn::generate_random_element(a, rng);
    element ret;

    mul(ret, a, fgkcn::zero_v);

    REQUIRE(ret == fgkcn::zero_v);
  }

  SECTION("of one with itself returns one") {
    constexpr element one{fgkb::r_v.data()};
    element ret;

    mul(ret, one, one);

    REQUIRE(one == ret);
  }

  SECTION("of pre-computed values returns expected value") {
    // Random Grumpkin base field element generated using the SAGE library.
    element a = 0x229385b6f3293f3646e2983182d91281f2f74a46ec96e25e9a6ebbaad85225f3_fgk;
    element b = 0x2e9503fd6a1e68cc189d9975664c2ce4f1e8cde24ed7b3666a3ec40f6661667b_fgk;
    element expected = 0x8cc1965a7cd255d59bb9c710d850c3ea5765dcddf21646694c8f38740811249_fgk;

    element ret;

    mul(ret, a, b);

    REQUIRE(expected == ret);
  }
}
