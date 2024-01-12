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
#include "sxt/field25/operation/sub.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/field25/constant/zero.h"
#include "sxt/field25/random/element.h"
#include "sxt/field25/type/element.h"

using namespace sxt;
using namespace sxt::f25o;

TEST_CASE("subtraction") {
  SECTION("of a random field element and zero returns the random field element") {
    f25t::element a;
    basn::fast_random_number_generator rng{1, 2};
    f25rn::generate_random_element(a, rng);

    f25t::element ret;

    sub(ret, a, f25cn::zero_v);

    REQUIRE(a == ret);
  }

  SECTION("of pre-computed values returns expected value") {
    // Random bn254 base field element generated using the SAGE library.
    constexpr f25t::element a{0x5277392d1dd1c9b6, 0xd4128bfefd8e4cf5, 0xa7d5b7f3662a0ee9,
                              0x0994b748e9d2dbe7};
    constexpr f25t::element b{0x9412b919732aa7c6, 0x714691531e314d76, 0xcf45bbcd6cf10aa2,
                              0x1295b5038773df78};
    constexpr f25t::element expected{0xfa850c2a83241f37, 0xfa4d653d47ceca0b, 0x90e041dc7aba5ca4,
                                     0x276350b843909c98};
    f25t::element ret;

    sub(ret, a, b);

    REQUIRE(expected == ret);
  }

  SECTION("of zero and one returns the modulus minus one") {
    constexpr f25t::element b{1, 0, 0, 0};
    constexpr f25t::element expected{0x3c208c16d87cfd46, 0x97816a916871ca8d, 0xb85045b68181585d,
                                     0x30644e72e131a029};
    f25t::element ret;

    sub(ret, f25cn::zero_v, b);

    REQUIRE(expected == ret);
  }
}
