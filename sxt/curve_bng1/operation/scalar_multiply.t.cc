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
#include "sxt/curve_bng1/operation/scalar_multiply.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve_bng1/constant/generator.h"
#include "sxt/curve_bng1/operation/add.h"
#include "sxt/curve_bng1/operation/double.h"
#include "sxt/curve_bng1/type/element_p2.h"

using namespace sxt;
using namespace sxt::cn1o;

TEST_CASE("scalar multiplication returns") {
  SECTION("the identity if the scalar is zero") {
    constexpr std::array<uint8_t, 32> a{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                        0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
    cn1t::element_p2 ret;

    scalar_multiply255(ret, cn1cn::generator_p2_v, a.data());

    REQUIRE(cn1t::element_p2::identity() == ret);
  }

  SECTION("the same value if the scalar is one") {
    constexpr std::array<uint8_t, 32> a{0x01, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                        0x0,  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                        0x0,  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
    cn1t::element_p2 ret;

    scalar_multiply255(ret, cn1cn::generator_p2_v, a.data());

    REQUIRE(cn1cn::generator_p2_v == ret);
  }

  SECTION("2G if the scalar is two") {
    constexpr std::array<uint8_t, 32> a{0x02, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                        0x0,  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                        0x0,  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};

    cn1t::element_p2 ret_scalar;
    scalar_multiply255(ret_scalar, cn1cn::generator_p2_v, a.data());

    cn1t::element_p2 g_2_add;
    add(g_2_add, cn1cn::generator_p2_v, cn1cn::generator_p2_v);

    cn1t::element_p2 g_2_double;
    double_element(g_2_double, cn1cn::generator_p2_v);

    REQUIRE(ret_scalar == g_2_add);
    REQUIRE(ret_scalar == g_2_double);
    REQUIRE(ret_scalar != cn1t::element_p2::identity());
  }

  SECTION("4G if the scalar is four") {
    constexpr std::array<uint8_t, 32> a{0x04, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                        0x0,  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                                        0x0,  0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};

    cn1t::element_p2 ret_scalar;
    scalar_multiply255(ret_scalar, cn1cn::generator_p2_v, a.data());

    cn1t::element_p2 g_2;
    cn1t::element_p2 g_4;
    add(g_2, cn1cn::generator_p2_v, cn1cn::generator_p2_v);
    double_element(g_4, g_2);

    REQUIRE(ret_scalar == g_4);
    REQUIRE(ret_scalar != cn1t::element_p2::identity());
  }

  SECTION("the scalar modulus to return zero") {
    constexpr std::array<uint8_t, 32> a{0x01, 0x00, 0x00, 0xf0, 0x93, 0xf5, 0xe1, 0x43,
                                        0x91, 0x70, 0xb9, 0x79, 0x48, 0xe8, 0x33, 0x28,
                                        0x5d, 0x58, 0x81, 0x81, 0xb6, 0x45, 0x50, 0xb8,
                                        0x29, 0xa0, 0x31, 0xe1, 0x72, 0x4e, 0x64, 0x30};
    cn1t::element_p2 ret;

    scalar_multiply255(ret, cn1cn::generator_p2_v, a.data());

    REQUIRE(cn1t::element_p2::identity() == ret);
  }
}
