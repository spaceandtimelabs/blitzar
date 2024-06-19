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
#include "sxt/fieldgk/operation/pow_vartime.h"

#include <iostream>

#include "sxt/base/test/unit_test.h"
#include "sxt/fieldgk/type/element.h"

using namespace sxt;
using namespace sxt::fgko;

TEST_CASE("pow_varitime") {
  SECTION("of pre-computed values returns expected value") {
    constexpr fgkt::element a{0xb92e567e0e2f6f1e, 0xc1d6653f0e1b09b, 0x6e52b0b322cdbd45,
                              0x18c27d738c4b7477};
    constexpr fgkt::element b{0x3c208c16d87cfd45, 0x97816a916871ca8d, 0xb85045b68181585d,
                              0x30644e72e131a029};
    constexpr fgkt::element expected{0xb513f9d751f64c03, 0x3f7b2093dcebd1f, 0x42276bfa13f711b8,
                                     0x10b1e99cf9afc988};
    fgkt::element ret;

    pow_vartime(ret, a, b);

    REQUIRE(expected == ret);
  }
}
