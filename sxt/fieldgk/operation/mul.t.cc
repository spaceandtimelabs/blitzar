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

using namespace sxt;
using namespace sxt::fgko;

TEST_CASE("multiplication") {
  SECTION("of a random field element and zero returns zero") {
    fgkt::element a;
    basn::fast_random_number_generator rng{1, 2};
    fgkrn::generate_random_element(a, rng);
    fgkt::element ret;

    mul(ret, a, fgkcn::zero_v);

    REQUIRE(ret == fgkcn::zero_v);
  }

  SECTION("of one with itself returns one") {
    constexpr fgkt::element one{fgkb::r_v.data()};
    fgkt::element ret;

    mul(ret, one, one);

    REQUIRE(one == ret);
  }

  SECTION("of pre-computed values returns expected value") {
    // Random bn254 base field element generated using the SAGE library.
    constexpr fgkt::element a{0x5d19786fd5f59520, 0xa80980aec93386cf, 0x6f49109c5fb69712,
                              0x19803c04269ae364};
    constexpr fgkt::element b{0x0746f2e0932048bf, 0xc05bea62ab71831e, 0x1342f6ebbc9497e9,
                              0x12b50c2429b1a851};
    constexpr fgkt::element expected{0x241cac338ef8e513, 0x98220ca91953d8c1, 0x0bf0a5fd342762a0,
                                     0x0e616e612ed86a67};
    fgkt::element ret;

    mul(ret, a, b);

    REQUIRE(expected == ret);
  }
}
