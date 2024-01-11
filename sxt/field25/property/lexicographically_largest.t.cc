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
#include "sxt/field12/property/lexicographically_largest.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field12/constant/one.h"
#include "sxt/field12/constant/zero.h"
#include "sxt/field12/type/element.h"

using namespace sxt;
using namespace sxt::f12p;

TEST_CASE("lexicographically largest correctly identifies") {
  SECTION("zero is not largest") { REQUIRE(!lexicographically_largest(f12cn::zero_v)); }

  SECTION("one is not largest") { REQUIRE(!lexicographically_largest(f12cn::one_v)); }

  SECTION("pre-computed value that is not largest") {
    constexpr f12t::element e{0xa1fafffffffe5557, 0x995bfff976a3fffe, 0x03f41d24d174ceb4,
                              0xf6547998c1995dbd, 0x778a468f507a6034, 0x020559931f7f8103};
    REQUIRE(!lexicographically_largest(e));
  }

  SECTION("pre-computed value that is largest") {
    constexpr f12t::element e{0x1804000000015554, 0x855000053ab00001, 0x633cb57c253c276f,
                              0x6e22d1ec31ebb502, 0xd3916126f2d14ca2, 0x17fbb8571a006596};
    REQUIRE(lexicographically_largest(e));
  }

  SECTION("another pre-computed value that is largest") {
    constexpr f12t::element e{0x43f5fffffffcaaae, 0x32b7fff2ed47fffd, 0x07e83a49a2e99d69,
                              0xeca8f3318332bb7a, 0xef148d1ea0f4c069, 0x040ab3263eff0206};
    REQUIRE(lexicographically_largest(e));
  }
}
