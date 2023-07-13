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
#include "sxt/field12/operation/pow_vartime.h"

#include <iostream>

#include "sxt/base/test/unit_test.h"
#include "sxt/field12/type/element.h"

using namespace sxt;
using namespace sxt::f12o;

TEST_CASE("pow_varitime") {
  SECTION("of pre-computed values returns expected value") {
    constexpr f12t::element a{0xaa270000000cfff3, 0x53cc0032fc34000a, 0x478fe97a6b0a807f,
                              0xb1d37ebee6ba24d7, 0x8ec9733bbf78ab2f, 0x09d645513d83de7e};
    constexpr f12t::element b{0xee7fbfffffffeaab, 0x07aaffffac54ffff, 0xd9cc34a83dac3d89,
                              0xd91dd2e13ce144af, 0x92c6e9ed90d2eb35, 0x0680447a8e5ff9a6};
    constexpr f12t::element expected{0x87ebfffffff9555c, 0x656fffe5da8ffffa, 0xfd0749345d33ad2,
                                     0xd951e663066576f4, 0xde291a3d41e980d3, 0x815664c7dfe040d};
    f12t::element ret;

    pow_vartime(ret, a, b);

    REQUIRE(expected == ret);
  }
}
