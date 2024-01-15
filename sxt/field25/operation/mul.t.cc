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
#include "sxt/field25/operation/mul.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/field25/base/constants.h"
#include "sxt/field25/base/montgomery.h"
#include "sxt/field25/constant/zero.h"
#include "sxt/field25/random/element.h"
#include "sxt/field25/type/element.h"

using namespace sxt;
using namespace sxt::f25o;

TEST_CASE("multiplication") {
  SECTION("of a random field element and zero returns zero") {
    f25t::element a;
    basn::fast_random_number_generator rng{1, 2};
    f25rn::generate_random_element(a, rng);
    f25t::element ret;

    mul(ret, a, f25cn::zero_v);

    REQUIRE(ret == f25cn::zero_v);
  }

  SECTION("of one with itself returns one") {
    constexpr f25t::element one{f25b::r_v.data()};
    f25t::element ret;

    mul(ret, one, one);

    REQUIRE(one == ret);
  }

  SECTION("of pre-computed values returns expected value") {
    // Random bn254 base field element generated using the SAGE library.
    constexpr f25t::element a{0x5d19786fd5f59520, 0xa80980aec93386cf, 0x6f49109c5fb69712,
                              0x19803c04269ae364};
    constexpr f25t::element b{0x0746f2e0932048bf, 0xc05bea62ab71831e, 0x1342f6ebbc9497e9,
                              0x12b50c2429b1a851};
    constexpr f25t::element expected{0x241cac338ef8e513, 0x98220ca91953d8c1, 0x0bf0a5fd342762a0,
                                     0x0e616e612ed86a67};
    f25t::element ret;

    mul(ret, a, b);

    REQUIRE(expected == ret);
  }
}
