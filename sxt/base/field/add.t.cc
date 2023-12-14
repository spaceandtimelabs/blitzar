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
#include "sxt/base/field/add.h"

#include "sxt/base/field/example_element.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/field12/constant/zero.h"
#include "sxt/field12/type/element.h"

using namespace sxt;
using namespace sxt::basfld;

TEST_CASE("addition") {
  SECTION("works on basic elements") {
    constexpr basfld::element1 a{6};
    constexpr basfld::element1 b{14};
    constexpr basfld::element1 exp{20};
    basfld::element1 ret;

    add(ret, a, b);

    REQUIRE(ret == exp);
  }

  SECTION("respects the modulus") {
    constexpr basfld::element1 a{90};
    constexpr basfld::element1 b{10};
    constexpr basfld::element1 exp{3};
    basfld::element1 ret;

    add(ret, a, b);

    REQUIRE(ret == exp);
  }
}

TEST_CASE("bls12-381 base field element addition") {
  SECTION("between pre-computed value and zero returns pre-computed value") {
    // Random values between 1 and the modulus generated using the SAGE library.
    constexpr f12t::element a{0xa18bb998ebd63ed4, 0xf9376579bd7313fb, 0x7c8e20c46f387e01,
                              0xfd64e34f83253657, 0x69f877012e12b25a, 0x9e91aa07f8a1e24};

    f12t::element ret;
    add(ret, a, f12cn::zero_v);

    REQUIRE(a == ret);
  }

  SECTION("between pre-computed values returns expected value") {
    // Random values between 1 and modulus generated using the SAGE library.
    constexpr f12t::element a{0xd13cd59bb4634e05, 0x0f2509b85592e56f, 0x651937e06008a619,
                              0x64d5bd872b39c8a7, 0x841f0f8892fa4f10, 0x78048ec7ecc6399};
    constexpr f12t::element b{0x16c9e69b4dea3f04, 0xf0a6ec99d0b1f9be, 0x22bb437b9b63365f,
                              0x1c46c6dd44489804, 0x10f8e4ed03d5659a, 0x14fe816ddb9e6192};
    constexpr f12t::element expected{0x2e07bc37024de25e, 0xe11ff65374f0df2e, 0x20a3a8bb04bae654,
                                     0x1ca538df7bfd4dec, 0x49fc4cbf538407d3, 0x27db87020eade91};

    f12t::element ret;
    add(ret, a, b);

    REQUIRE(expected == ret);
  }

  SECTION("between the modulus minus one and one returns zero") {
    constexpr f12t::element a{0xb9feffffffffaaaa, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624,
                              0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a};
    constexpr f12t::element b{0x1, 0x0, 0x0, 0x0, 0x0, 0x0};

    f12t::element ret;
    add(ret, a, b);

    REQUIRE(f12cn::zero_v == ret);
  }
}
