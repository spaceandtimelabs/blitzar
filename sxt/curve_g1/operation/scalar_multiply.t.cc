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
#include "sxt/curve_g1/operation/scalar_multiply.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve_g1/constant/generator.h"
#include "sxt/curve_g1/constant/identity.h"
#include "sxt/curve_g1/type/element_p2.h"

using namespace sxt;
using namespace sxt::cg1o;

TEST_CASE("scalar multiplication returns") {
  SECTION("the identity if the scalar is zero") {
    constexpr std::array<uint8_t, 32> a{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
    cg1t::element_p2 ret;

    scalar_multiply255(ret, cg1cn::generator_p2_v, a.data());

    REQUIRE(cg1cn::identity_p2_v == ret);
  }

  SECTION("the identity if only the first bit is 1") {
    constexpr std::array<uint8_t, 32> a{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x80};
    cg1t::element_p2 ret;

    scalar_multiply255(ret, cg1cn::generator_p2_v, a.data());

    REQUIRE(cg1cn::identity_p2_v == ret);
  }

  SECTION("the same value if the scalar is one") {
    constexpr std::array<uint8_t, 32> a{0x01, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                        0x0,  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                        0x0,  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
    cg1t::element_p2 ret;

    scalar_multiply255(ret, cg1cn::generator_p2_v, a.data());

    REQUIRE(cg1cn::generator_p2_v == ret);
  }

  SECTION("the expected pre-computed value") {
    constexpr std::array<uint8_t, 32> a{0x1b, 0xa7, 0x6d, 0xa5, 0x98, 0x82, 0x56, 0x2b,
                                        0xd2, 0x19, 0xf5, 0xe,  0xc8, 0xfa, 0x5,  0x85,
                                        0x91, 0xe7, 0x1d, 0x5e, 0xd2, 0x60, 0x22, 0x10,
                                        0x6a, 0xdc, 0x18, 0xfd, 0xfc, 0xf8, 0x9a, 0xc};

    constexpr cg1t::element_p2 expected{{0xa80565e509d658c5, 0xff0b490d2a6da917, 0xf178d3cd7d4ff503,
                                         0xbb1dcbc2d53ddf89, 0x6d4bca121da390f7, 0xa40068d3eaaef89},
                                        {0x6fae81a2c5d9320f, 0x127185de1966cef9, 0xfa1a4697641b53ef,
                                         0x59db1d37761ab56d, 0xebc9e60da366ab0f, 0x9574ab82b0a5f75},
                                        {0xb25fa910e5757174, 0xf51ff09386bdbb6a, 0x5a603696e84b08ff,
                                         0x6bcedba19d0c2e30, 0xc1fe7e38dfd460cf,
                                         0xb6f9f61a7f4a36e}};
    cg1t::element_p2 ret;

    scalar_multiply255(ret, cg1cn::generator_p2_v, a.data());

    REQUIRE(expected == ret);
  }
}
