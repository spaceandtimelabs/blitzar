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
#include "sxt/fieldgk/constant/zero.h"
#include "sxt/fieldgk/type/element.h"

using namespace sxt;
using namespace sxt::fgko;

TEST_CASE("negation") {
  SECTION("of zero and the modulus are equal") {
    fgkt::element ret_zero;
    fgkt::element ret_modulus;

    neg(ret_zero, fgkcn::zero_v);
    neg(ret_modulus, fgkb::p_v.data());

    REQUIRE(ret_zero == ret_modulus);
  }

  SECTION("of the modulus minus one is one") {
    constexpr fgkt::element modulus_minus_one{0x3c208c16d87cfd46, 0x97816a916871ca8d,
                                              0xb85045b68181585d, 0x30644e72e131a029};
    constexpr fgkt::element one{1, 0, 0, 0};
    fgkt::element ret;

    neg(ret, modulus_minus_one);

    REQUIRE(ret == one);
  }

  SECTION("of a pre-computed value is expected") {
    // Random bn254 base field element generated using the SAGE library.
    constexpr fgkt::element a{0x1149c21473b043fd, 0x5610b2a5c08c7ecf, 0xc9e31f2d914c45b5,
                              0x066031eb7a3ca7fd};
    constexpr fgkt::element expected{0x2ad6ca0264ccb94a, 0x4170b7eba7e54bbe, 0xee6d2688f03512a8,
                                     0x2a041c8766f4f82b};
    fgkt::element ret;

    neg(ret, a);

    REQUIRE(expected == ret);
  }
}
