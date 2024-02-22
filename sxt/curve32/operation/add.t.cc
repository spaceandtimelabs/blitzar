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
#include "sxt/curve32/operation/add.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve32/type/element_p3.h"

using namespace sxt;
using namespace sxt::c32o;

TEST_CASE("we can add curve elements") {
  c32t::element_p3 g{
      {3990542415680775, 3398198340507945, 4322667446711068, 2814063955482877, 2839572215813860},
      {1801439850948184, 1351079888211148, 450359962737049, 900719925474099, 1801439850948198},
      {1, 0, 0, 0, 0},
      {1841354044333475, 16398895984059, 755974180946558, 900171276175154, 1821297809914039},
  };

  c32t::element_p3 res;

  SECTION("we can add the identity") {
    add(res, c32t::element_p3{c32t::element_p3::identity()}, g);
    REQUIRE(res == g);
  }

  SECTION("we can add inplace arguments") {
    c32t::element_p3 lhs{g};
    c32t::element_p3 rhs = c32t::element_p3::identity();
    add_inplace(lhs, rhs);
    REQUIRE(lhs == g);
  }
}
