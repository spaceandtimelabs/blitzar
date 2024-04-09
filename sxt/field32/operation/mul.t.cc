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
#include "sxt/field32/operation/mul.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/field32/base/constants.h"
#include "sxt/field32/base/montgomery.h"
#include "sxt/field32/constant/zero.h"
#include "sxt/field32/random/element.h"
#include "sxt/field32/type/element.h"

using namespace sxt;
using namespace sxt::f32o;

TEST_CASE("multiplication") {
  SECTION("of a random field element and zero returns zero") {
    f32t::element a;
    basn::fast_random_number_generator rng{1, 2};
    f32rn::generate_random_element(a, rng);
    f32t::element ret;

    mul(ret, a, f32cn::zero_v);

    REQUIRE(ret == f32cn::zero_v);
  }

  SECTION("of one with itself returns one") {
    constexpr f32t::element one{f32b::r_v.data()};
    f32t::element ret;

    mul(ret, one, one);

    REQUIRE(one == ret);
  }

  SECTION("of pre-computed values returns expected value") {
    // Random bn254 base field element generated using the SAGE library.
    constexpr f32t::element a{0xd5f59520, 0x5d19786f, 0xc93386cf, 0xa80980ae,
                              0x5fb69712, 0x6f49109c, 0x269ae364, 0x19803c04};
    constexpr f32t::element b{0x932048bf, 0x0746f2e0, 0xab71831e, 0xc05bea62,
                              0xbc9497e9, 0x1342f6eb, 0x29b1a851, 0x12b50c24};
    constexpr f32t::element expected{0x8ef8e513, 0x241cac33, 0x1953d8c1, 0x98220ca9,
                                     0x342762a0, 0x0bf0a5fd, 0x2ed86a67, 0x0e616e61};
    f32t::element ret;

    mul(ret, a, b);

    REQUIRE(expected == ret);
  }
}
