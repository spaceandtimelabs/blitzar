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
#include "sxt/field32/operation/pow_vartime.h"

#include <iostream>

#include "sxt/base/test/unit_test.h"
#include "sxt/field32/type/element.h"

using namespace sxt;
using namespace sxt::f32o;

TEST_CASE("pow_varitime") {
  SECTION("of pre-computed values returns expected value") {
    constexpr f32t::element a{0x0e2f6f1e, 0xb92e567e, 0xf0e1b09b, 0x0c1d6653,
                              0x22cdbd45, 0x6e52b0b3, 0x8c4b7477, 0x18c27d73};
    constexpr f32t::element b{0xd87cfd45, 0x3c208c16, 0x6871ca8d, 0x97816a91,
                              0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
    constexpr f32t::element expected{0x51f64c03, 0xb513f9d7, 0x3dcebd1f, 0x03f7b209,
                                     0x13f711b8, 0x42276bfa, 0xf9afc988, 0x10b1e99c};
    f32t::element ret;

    pow_vartime(ret, a, b);

    REQUIRE(expected == ret);
  }
}
