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
#include "sxt/field32/operation/add.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/field32/constant/zero.h"
#include "sxt/field32/random/element.h"
#include "sxt/field32/type/element.h"

using namespace sxt;
using namespace sxt::f32o;

TEST_CASE("addition") {
  SECTION("of a random field element and zero returns the random field element") {
    f32t::element a;
    basn::fast_random_number_generator rng{1, 2};
    f32rn::generate_random_element(a, rng);

    f32t::element ret;

    add(ret, a, f32cn::zero_v);

    REQUIRE(a == ret);
  }

  SECTION("of pre-computed values returns expected value") {
    // Random bn254 base field elements generated using the SAGE library.
    constexpr f32t::element a{0x73b043fd, 0x1149c214, 0xc08c7ecf, 0x5610b2a5,
                              0x914c45b5, 0xc9e31f2d, 0x7a3ca7fd, 0x066031eb};
    constexpr f32t::element b{0x60d431b8, 0x13f757e6, 0x237b60d5, 0x8a86bc6a,
                              0x22e9b96d, 0x6f91e115, 0x724f624b, 0x10ce4233};
    constexpr f32t::element expected{0xd48475b5, 0x254119fa, 0xe407dfa4, 0xe0976f0f,
                                     0xb435ff22, 0x39750042, 0xec8c0a49, 0x172e741e};
    f32t::element ret;

    add(ret, a, b);

    REQUIRE(expected == ret);
  }

  SECTION("of a pre-computed value the modulus minus one returns expected value") {
    // Random bn254 base field element generated using the SAGE library.
    constexpr f32t::element a{0x73b043fd, 0x1149c214, 0xc08c7ecf, 0x5610b2a5,
                              0x914c45b5, 0xc9e31f2d, 0x7a3ca7fd, 0x066031eb};
    constexpr f32t::element b{0xd87cfd46, 0x3c208c16, 0x6871ca8d, 0x97816a91,
                              0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
    constexpr f32t::element expected{0x73b043fc, 0x1149c214, 0xc08c7ecf, 0x5610b2a5,
                                     0x914c45b5, 0xc9e31f2d, 0x7a3ca7fd, 0x066031eb};
    f32t::element ret;

    add(ret, a, b);

    REQUIRE(expected == ret);
  }

  SECTION("of the modulus minus one and one returns zero") {
    constexpr f32t::element a{0xd87cfd46, 0x3c208c16, 0x6871ca8d, 0x97816a91,
                              0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
    constexpr f32t::element b{1, 0, 0, 0, 0, 0, 0, 0};
    f32t::element ret;

    add(ret, a, b);

    REQUIRE(f32cn::zero_v == ret);
  }
}
