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
#include "sxt/fieldgk/operation/add.h"

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

TEST_CASE("addition") {
  SECTION("of a random field element and zero returns the random field element") {
    element a;
    basn::fast_random_number_generator rng{1, 2};
    fgkrn::generate_random_element(a, rng);

    element ret;

    add(ret, a, fgkcn::zero_v);

    REQUIRE(a == ret);
  }

  SECTION("of pre-computed values returns expected value") {
    // Random Grumpkin base field elements generated using the SAGE library.
    element a = 0x11d00c6953e75d5492d56f7f4902850ea43757d7568ac00b6462df193faa5a05_fgk;
    element b = 0x17417d1a331383d6cfd62b81d4efacf24a1fe95a261bdadc86f4216399a0e113_fgk;
    element expected = 0x2911898386fae12b62ab9b011df23200ee5741317ca69ae7eb57007cd94b3b18_fgk;

    element ret;

    add(ret, a, b);

    REQUIRE(expected == ret);
  }

  SECTION("of a pre-computed value the modulus minus one returns expected value") {
    // Random Grumpkin base field element generated using the SAGE library.
    element a = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000_fgk;
    element b = 0x0758e1cf42f87142597d510b7d269165975d7f48c648b216942a29c3adacca0d_fgk;
    element expected = 0x0758e1cf42f87142597d510b7d269165975d7f48c648b216942a29c3adacca0c_fgk;

    element ret;

    add(ret, a, b);

    REQUIRE(expected == ret);
  }

  SECTION("of the modulus minus one and one returns zero") {
    element a = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000_fgk;
    element b = fgkcn::one_v;
    element ret;

    add(ret, a, b);

    REQUIRE(fgkcn::zero_v == ret);
  }
}
