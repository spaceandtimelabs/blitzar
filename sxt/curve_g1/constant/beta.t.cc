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
/*
 * Adopted from zkcrypto/bls12_381
 *
 * Copyright (c) 2021
 * Sean Bowe <ewillbefull@gmail.com>
 * Jack Grigg <thestr4d@gmail.com>
 *
 * See third_party/license/zkcrypto.LICENSE
 */
#include "sxt/curve_g1/constant/beta.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field12/base/byte_conversion.h"
#include "sxt/field12/constant/one.h"
#include "sxt/field12/operation/mul.h"
#include "sxt/field12/type/element.h"

using namespace sxt;
using namespace sxt::cg1cn;

TEST_CASE("the beta constant") {
  SECTION("should equal the third root of unity") {
    REQUIRE(cg1cn::beta_v != f12cn::one_v);

    f12t::element beta_squared;
    f12o::mul(beta_squared, cg1cn::beta_v, cg1cn::beta_v);
    REQUIRE(beta_squared != f12cn::one_v);

    f12t::element beta_cubed;
    f12o::mul(beta_cubed, beta_squared, cg1cn::beta_v);
    REQUIRE(beta_cubed == f12cn::one_v);
  }

  SECTION("should be equal to the following little endian byte representation") {
    bool is_below_modulus{true};
    std::array<uint64_t, 6> h{};
    constexpr std::array<uint8_t, 48> s{0xfe, 0xff, 0xfe, 0xff, 0xff, 0xff, 0x01, 0x2e, 0x02, 0x00,
                                        0x0a, 0x62, 0x13, 0xd8, 0x17, 0xde, 0x88, 0x96, 0xf8, 0xe6,
                                        0x3b, 0xa9, 0xb3, 0xdd, 0xea, 0x77, 0x0f, 0x6a, 0x07, 0xc6,
                                        0x69, 0xba, 0x51, 0xce, 0x76, 0xdf, 0x2f, 0x67, 0x19, 0x5f,
                                        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

    f12b::from_bytes(is_below_modulus, h.data(), s.data());

    const f12t::element expected{h[0], h[1], h[2], h[3], h[4], h[5]};

    REQUIRE(expected == cg1cn::beta_v);
  }
}
