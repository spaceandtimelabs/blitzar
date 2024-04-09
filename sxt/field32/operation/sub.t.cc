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
#include "sxt/field32/operation/sub.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/field32/constant/zero.h"
#include "sxt/field32/random/element.h"
#include "sxt/field32/type/element.h"

using namespace sxt;
using namespace sxt::f32o;

TEST_CASE("subtraction") {
  SECTION("of a random field element and zero returns the random field element") {
    f32t::element a;
    basn::fast_random_number_generator rng{1, 2};
    f32rn::generate_random_element(a, rng);

    f32t::element ret;

    sub(ret, a, f32cn::zero_v);

    REQUIRE(a == ret);
  }

  SECTION("of pre-computed values returns expected value") {
    // Random bn254 base field element generated using the SAGE library.
    constexpr f32t::element a{0x1dd1c9b6, 0x5277392d, 0xfd8e4cf5, 0xd4128bfe,
                              0x662a0ee9, 0xa7d5b7f3, 0xe9d2dbe7, 0x0994b748};
    constexpr f32t::element b{0x732aa7c6, 0x9412b919, 0x1e314d76, 0x71469153,
                              0x6cf10aa2, 0xcf45bbcd, 0x8773df78, 0x1295b503};
    constexpr f32t::element expected{0x83241f37, 0xfa850c2a, 0x47ceca0b, 0xfa4d653d,
                                     0x7aba5ca4, 0x90e041dc, 0x43909c98, 0x276350b8};
    f32t::element ret;

    sub(ret, a, b);

    REQUIRE(expected == ret);
  }

  SECTION("of zero and one returns the modulus minus one") {
    constexpr f32t::element b{1, 0, 0, 0, 0, 0, 0, 0};
    constexpr f32t::element expected{0xd87cfd46, 0x3c208c16, 0x6871ca8d, 0x97816a91,
                                     0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
    f32t::element ret;

    sub(ret, f32cn::zero_v, b);

    REQUIRE(expected == ret);
  }
}
