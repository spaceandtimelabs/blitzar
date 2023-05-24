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
#include "sxt/curve21/operation/add.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/type/element_p3.h"

using namespace sxt;
using namespace sxt::c21o;

TEST_CASE("we can add curve elements") {
  c21t::element_p3 g{
      .X{3990542415680775, 3398198340507945, 4322667446711068, 2814063955482877, 2839572215813860},
      .Y{1801439850948184, 1351079888211148, 450359962737049, 900719925474099, 1801439850948198},
      .Z{1, 0, 0, 0, 0},
      .T{1841354044333475, 16398895984059, 755974180946558, 900171276175154, 1821297809914039},
  };

  c21t::element_p3 res;

  SECTION("adding zero acts as the identity function") {
    add(res, c21t::element_p3{c21cn::zero_p3_v}, g);
    REQUIRE(res == g);
  }

  SECTION("we can add inplace arguments") {
    c21t::element_p3 lhs{g};
    c21t::element_p3 rhs = c21cn::zero_p3_v;
    add_inplace(lhs, rhs);
    REQUIRE(lhs == g);
  }

  SECTION("we can add volatile arguments") {
    volatile c21t::element_p3 lhs{g};
    volatile c21t::element_p3 rhs = c21cn::zero_p3_v;
    add_inplace(lhs, rhs);
    res = {
        .X{lhs.X},
        .Y{lhs.Y},
        .Z{lhs.Z},
        .T{lhs.T},
    };
    REQUIRE(res == g);
  }
}
