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
#include "sxt/field12/operation/invert.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field12/constant/one.h"
#include "sxt/field12/operation/mul.h"
#include "sxt/field12/type/element.h"

using namespace sxt;
using namespace sxt::f12o;

TEST_CASE("inversion") {
  SECTION("of a pre-computed value multiplied by its inverse is equal to one in Montogomery form") {
    constexpr f12t::element a{0x4fb9be8a496563c1, 0x98a27a4674c765e1, 0x8475a3d548b133fe,
                              0xdbeb89b089f08299, 0x5c1fb7a3c186887f, 0x13db2e403f66ba22};

    f12t::element a_inv;
    auto is_zero = invert(a_inv, a);
    REQUIRE(!is_zero);

    f12t::element ret_mul;
    mul(ret_mul, a, a_inv);
    REQUIRE(ret_mul == f12cn::one_v);
  }

  SECTION("of pre-computed values returns expected value") {
    constexpr f12t::element a{0x43b43a5078ac2076, 0x1ce0763046f8962b, 0x724a5276486d735c,
                              0x6f05c2a6282d48fd, 0x2095bd5bb4ca9331, 0x03b35b3894b0f7da};
    constexpr f12t::element expected{0x69ecd7040952148f, 0x985ccc2022190f55, 0xe19bba36a9ad2f41,
                                     0x19bb16c95219dbd8, 0x14dcacfdfb478693, 0x115ff58afff9a8e1};

    f12t::element ret;
    auto is_zero = invert(ret, a);

    REQUIRE(!is_zero);
    REQUIRE(expected == ret);
  }

  SECTION("of zero returns the a flag indicating the value is zero") {
    constexpr f12t::element a{0x0, 0x0, 0x0, 0x0, 0x0, 0x0};

    f12t::element ret;
    auto is_zero = invert(ret, a);

    REQUIRE(is_zero);
  }
}
