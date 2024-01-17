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
#include "sxt/curve_bng1/type/element_affine.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field25/constant/one.h"
#include "sxt/field25/constant/zero.h"

using namespace sxt;
using namespace sxt::cn1t;

TEST_CASE("affine element equality") {
  SECTION("can distinguish the generator from the identity") {
    constexpr element_affine a{{0x5cb38790fd530c16, 0x7817fc679976fff5, 0x154f95c7143ba1c1,
                                0xf0ae6acdf3d0e747, 0xedce6ecc21dbf440, 0x120177419e0bfb75},
                               {0xbaac93d50ce72271, 0x8c22631a7918fd8e, 0xdd595f13570725ce,
                                0x51ac582950405194, 0x0e1c8c3fad0059c0, 0x0bbc3efc5008a26a},
                               false};

    constexpr element_affine b{f12cn::zero_v, f12cn::one_v, true};

    REQUIRE(a == a);
    REQUIRE(b == b);
    REQUIRE(a != b);
    REQUIRE(b != a);
  }
}
