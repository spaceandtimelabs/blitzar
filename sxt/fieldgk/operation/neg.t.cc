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
#include "sxt/fieldgk/operation/neg.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/fieldgk/base/constants.h"
#include "sxt/fieldgk/constant/one.h"
#include "sxt/fieldgk/constant/zero.h"
#include "sxt/fieldgk/type/element.h"
#include "sxt/fieldgk/type/literal.h"

using namespace sxt;
using namespace sxt::fgko;
using namespace sxt::fgkt;

TEST_CASE("negation") {
  SECTION("of zero and the modulus are equal") {
    element ret_zero;
    element ret_modulus;

    neg(ret_zero, fgkcn::zero_v);
    neg(ret_modulus, fgkb::p_v.data());

    REQUIRE(ret_zero == ret_modulus);
  }

  SECTION("of the modulus minus one is one") {
    element modulus_minus_one =
        0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000_fgk;
    element ret;

    neg(ret, modulus_minus_one);

    REQUIRE(ret == fgkcn::one_v);
  }

  SECTION("of a pre-computed value is expected") {
    // Random Grumpkin base field element generated using the SAGE library.
    element a = 0x11d00c6953e75d5492d56f7f4902850ea43757d7568ac00b6462df193faa5a05_fgk;
    element expected = 0x1e9442098d4a42d5257ad637387ed34e83fc9071232eb085df7f167ab055a5fc_fgk;

    element ret;

    neg(ret, a);

    REQUIRE(expected == ret);
  }
}
