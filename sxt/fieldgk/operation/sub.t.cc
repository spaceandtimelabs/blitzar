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
#include "sxt/fieldgk/operation/sub.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/fieldgk/constant/one.h"
#include "sxt/fieldgk/constant/zero.h"
#include "sxt/fieldgk/random/element.h"
#include "sxt/fieldgk/type/element.h"
#include "sxt/fieldgk/type/literal.h"

using namespace sxt;
using namespace sxt::fgko;
using namespace sxt::fgkt;

TEST_CASE("subtraction") {
  SECTION("of a random field element and zero returns the random field element") {
    fgkt::element a;
    basn::fast_random_number_generator rng{1, 2};
    fgkrn::generate_random_element(a, rng);

    fgkt::element ret;

    sub(ret, a, fgkcn::zero_v);

    REQUIRE(a == ret);
  }

  SECTION("of pre-computed values returns expected value") {
    // Random Grumpkin base field element generated using the SAGE library.
    element a = 0x160d1ee279a8b33aaad264b6b74af54dd1e8c7c51d515b66287e4cff880df122_fgk;
    element b = 0x249b84b7a5b659ee923591385e5e6370e5a91354f48eff9a234a35efe01e7ce0_fgk;
    element expected = 0x21d5e89db523f975d0ed1934da6dea3a14739cb8a27bcc5d49160ca397ef7443_fgk;
    element ret;

    sub(ret, a, b);

    REQUIRE(expected == ret);
  }

  SECTION("of zero and one returns the modulus minus one") {
    element expected = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000_fgk;
    element ret;

    sub(ret, fgkcn::zero_v, fgkcn::one_v);

    REQUIRE(expected == ret);
  }
}
