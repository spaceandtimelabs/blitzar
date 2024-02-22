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
#include "sxt/curve32/operation/scalar_multiply.h"

#include <iostream>

#include "sxt/base/test/unit_test.h"
#include "sxt/curve32/operation/add.h"
#include "sxt/curve32/type/element_p3.h"

using namespace sxt;
using namespace sxt::c32o;

TEST_CASE("we can multiply elements by a scalar") {
  c32t::element_p3 g{
      {3990542415680775, 3398198340507945, 4322667446711068, 2814063955482877, 2839572215813860},
      {1801439850948184, 1351079888211148, 450359962737049, 900719925474099, 1801439850948198},
      {1, 0, 0, 0, 0},
      {1841354044333475, 16398895984059, 755974180946558, 900171276175154, 1821297809914039},
  };

  c32t::element_p3 res;

  SECTION("verify multiply by 1") {
    unsigned char a[32] = {1};
    scalar_multiply255(res, a, g);
    REQUIRE(res.X == f51t::element{1987682947780000, 773294264508247, 919172218267419,
                                   891376861726595, 2032146878734994});
    REQUIRE(res.Y == f51t::element{1210076552040731, 1843649357132683, 227873395513447,
                                   2008892784295914, 2084244428540894});
    REQUIRE(res.Z == f51t::element{949645736629616, 1741611742994542, 1410741651234433,
                                   1385216073527268, 916455675412182});
    REQUIRE(res.T == f51t::element{239066470012855, 1969715299817747, 284977811876885,
                                   2064181377592425, 1175357540250945});
  }

  SECTION("verify multiply by 2") {
    unsigned char a[32] = {2};
    scalar_multiply255(res, a, g);
    REQUIRE(res.X == f51t::element{1002030577009317, 413985402962749, 1107992705352421,
                                   1701819753420419, 8688124941616});
    REQUIRE(res.Y == f51t::element{1961256026927373, 940468905986184, 259748503222505,
                                   924506438980976, 2169563456582984});
    REQUIRE(res.Z == f51t::element{663843209942905, 81365301039004, 1580407463606746,
                                   2087089281147264, 323194334940088});
    REQUIRE(res.T == f51t::element{1263134504727466, 2180639216875652, 658647461258916,
                                   1312984564731448, 533598071106974});
  }

  SECTION("verify we can multiply by an exponent with a[31] > 127") {
    uint8_t a1[32] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 125,
    };
    uint8_t a2[32] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 125,
    };
    uint8_t a3[32] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 250,
    };
    REQUIRE(a3[31] > 127);
    scalar_multiply255(res, a1, g);
    auto expected_res = res;

    scalar_multiply255(res, a2, g);
    c32o::add(expected_res, expected_res, res);

    scalar_multiply(res, basct::span<uint8_t>{a3, 32}, g);
    REQUIRE(res == expected_res);
  }
}
