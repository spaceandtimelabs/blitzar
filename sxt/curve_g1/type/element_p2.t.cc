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
/**
 * Adopted from zkcrypto/bls12_381
 *
 * Copyright (c) 2021
 * Sean Bowe <ewillbefull@gmail.com>
 * Jack Grigg <thestr4d@gmail.com>
 *
 * See third_party/license/zkcrypto.LICENSE
 */
#include "sxt/curve_g1/type/element_p2.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field12/constant/one.h"
#include "sxt/field12/constant/zero.h"
#include "sxt/field12/operation/mul.h"
#include "sxt/field12/type/element.h"
#include "sxt/field12/type/literal.h"

using namespace sxt;
using namespace sxt::cg1t;
using f12t::operator""_f12;

TEST_CASE("projective element equality") {
  SECTION("can distinguish the generator from the identity") {
    constexpr element_p2 a{{0x5cb38790fd530c16, 0x7817fc679976fff5, 0x154f95c7143ba1c1,
                            0xf0ae6acdf3d0e747, 0xedce6ecc21dbf440, 0x120177419e0bfb75},
                           {0xbaac93d50ce72271, 0x8c22631a7918fd8e, 0xdd595f13570725ce,
                            0x51ac582950405194, 0x0e1c8c3fad0059c0, 0x0bbc3efc5008a26a},
                           f12cn::one_v};
    constexpr element_p2 b{f12cn::zero_v, f12cn::one_v, f12cn::zero_v};

    REQUIRE(a == a);
    REQUIRE(b == b);
    REQUIRE(a != b);
    REQUIRE(b != a);

    constexpr f12t::element z{0xba7afa1f9a6fe250, 0xfa0f5b595eafe731, 0x3bdc477694c306e7,
                              0x2149be4b3949fa24, 0x64aa6e0649b2078c, 0x12b108ac33643c3e};

    f12t::element ax_z;
    f12o::mul(ax_z, a.X, z);

    f12t::element ay_z;
    f12o::mul(ay_z, a.Y, z);

    element_p2 c{ax_z, ay_z, z};

    REQUIRE(a == c);
    REQUIRE(b != c);
    REQUIRE(c == a);
    REQUIRE(c != b);
  }
}

TEST_CASE("we can convert between elements") {
  SECTION("we can convert the identity element") {
    auto id = element_p2::identity();
    auto id_p = element_p2{static_cast<compact_element>(id)};
    REQUIRE(id == id_p);
  }

  SECTION("we can covert an arbitrary element") {
    element_p2 e{
        0x17f7b262294ef7b666e940cecd80b68f5f84158fbb044c0e7f4c4fb15c4b58679609c912fd8648e9c121e09dfbc0141c_f12,
        0x2caf9ee971b3d3212363b9b76bdb02c3ec2736aef5e37dd205099ab55cc2b3950c3a01d27db55031ffc1872a669a8c0_f12,
        0x15ee88dd9836d9791f49e2a9702f040484db019beea103a5d68a4bce2e0fecc878970c1f2dc242111146f26916b000b6_f12};
    auto ep = element_p2{static_cast<compact_element>(e)};
    REQUIRE(e == ep);
  }
}
