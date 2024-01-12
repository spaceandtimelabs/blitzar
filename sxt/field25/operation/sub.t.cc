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

#include "sxt/base/test/unit_test.h"
#include "sxt/field25/constant/zero.h"
#include "sxt/field25/type/element.h"

using namespace sxt;
using namespace sxt::f25o;

TEST_CASE("subtraction") {
  SECTION("of pre-computed value and zero returns pre-computed value") {
    // Random values between 1 and p_v generated using the SAGE library.
    constexpr f12t::element a{0xd841a79826bf52bd, 0xf03e2c871b72b39f, 0x3e0849ec3694f61f,
                              0x9f2941b0ba71fbae, 0xdc01791735636ad0, 0x16049d32722cd303};

    f12t::element ret;
    sub(ret, a, f12cn::zero_v);

    REQUIRE(a == ret);
  }

  SECTION("of pre-computed values returns expected value") {
    // Random values between 1 and p_v generated using the SAGE library.
    constexpr f12t::element a{0x3a67dc4bfd86d199, 0x131d9f2a59475351, 0xa4277d44c833b67a,
                              0x21ea1c79f6aedd93, 0x3fadab0c84079145, 0x179af60bd96d6d51};
    constexpr f12t::element b{0xff30f3d15bd9e635, 0x9565c755ac959535, 0x836b6466e8730b6a,
                              0x9e9f141d4b78bb12, 0x265d74738f87c381, 0x0a3adb6a76b8e291};
    constexpr f12t::element expected{0x3b36e87aa1aceb64, 0x7db7d7d4acb1be1b, 0x20bc18dddfc0ab0f,
                                     0x834b085cab362281, 0x19503698f47fcdc3, 0xd601aa162b48ac0};

    f12t::element ret;
    sub(ret, a, b);

    REQUIRE(expected == ret);
  }

  SECTION("of zero and one returns the modulus minus one") {
    constexpr f12t::element b{0x1, 0x0, 0x0, 0x0, 0x0, 0x0};
    constexpr f12t::element expected{0xb9feffffffffaaaa, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624,
                                     0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a};

    f12t::element ret;
    sub(ret, f12cn::zero_v, b);

    REQUIRE(expected == ret);
  }
}
