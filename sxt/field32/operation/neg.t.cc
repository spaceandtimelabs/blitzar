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
#include "sxt/field32/operation/neg.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field32/base/constants.h"
#include "sxt/field32/constant/zero.h"
#include "sxt/field32/type/element.h"

using namespace sxt;
using namespace sxt::f32o;

TEST_CASE("negation") {
  SECTION("of zero and the modulus are equal") {
    f32t::element ret_zero;
    f32t::element ret_modulus;

    neg(ret_zero, f32cn::zero_v);
    neg(ret_modulus, f32b::p_v.data());

    REQUIRE(ret_zero == ret_modulus);
  }

  SECTION("of the modulus minus one is one") {
    constexpr f32t::element modulus_minus_one{0xd87cfd46, 0x3c208c16, 0x6871ca8d, 0x97816a91,
                                              0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
    constexpr f32t::element one{1, 0, 0, 0, 0, 0, 0, 0};
    f32t::element ret;

    neg(ret, modulus_minus_one);

    REQUIRE(ret == one);
  }

  SECTION("of a pre-computed value is expected") {
    // Random bn254 base field element generated using the SAGE library.
    constexpr f32t::element a{0x73b043fd, 0x1149c214, 0xc08c7ecf, 0x5610b2a5,
                              0x914c45b5, 0xc9e31f2d, 0x7a3ca7fd, 0x066031eb};
    constexpr f32t::element expected{0x64ccb94a, 0x2ad6ca02, 0xa7e54bbe, 0x4170b7eb,
                                     0xf03512a8, 0xee6d2688, 0x66f4f82b, 0x2a041c87};
    f32t::element ret;

    neg(ret, a);

    REQUIRE(expected == ret);
  }
}
