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
#include "sxt/field25/operation/neg.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field25/base/constants.h"
#include "sxt/field25/type/element.h"

using namespace sxt;
using namespace sxt::f25o;

TEST_CASE("negation") {
  SECTION("of zero and the modulus are equal") {
    constexpr f12t::element zero{0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
    constexpr f12t::element modulus{f12b::p_v.data()};
    f12t::element ret_zero;
    f12t::element ret_modulus;

    neg(ret_zero, zero);
    neg(ret_modulus, modulus);

    REQUIRE(ret_zero == ret_modulus);
  }

  SECTION("of the modulus minus one is one") {
    constexpr f12t::element modulus_minus_one{0xb9feffffffffaaaa, 0x1eabfffeb153ffff,
                                              0x6730d2a0f6b0f624, 0x64774b84f38512bf,
                                              0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a};
    constexpr f12t::element one{0x1, 0x0, 0x0, 0x0, 0x0, 0x0};
    f12t::element ret;

    neg(ret, modulus_minus_one);

    REQUIRE(ret == one);
  }

  SECTION("of a pre-computed value is expected") {
    constexpr f12t::element a{0x5360bb5978678032, 0x7dd275ae799e128e, 0x5c5b5071ce4f4dcf,
                              0xcdb21f93078dbb3e, 0xc32365c5e73f474a, 0x115a2a5489babe5b};
    constexpr f12t::element expected{0x669e44a687982a79, 0xa0d98a5037b5ed71, 0x0ad5822f2861a854,
                                     0x96c52bf1ebf75781, 0x87f841f05c0c658c, 0x08a6e795afc5283e};
    f12t::element ret;

    neg(ret, a);

    REQUIRE(expected == ret);
  }
}
