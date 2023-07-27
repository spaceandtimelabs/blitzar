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
#include "sxt/curve21/operation/double.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/type/conversion_utility.h"
#include "sxt/curve21/type/element_p3.h"

using namespace sxt;
using namespace sxt::c21o;

TEST_CASE("we can double curve elements") {
  c21t::element_p3 g{
      {3990542415680775, 3398198340507945, 4322667446711068, 2814063955482877, 2839572215813860},
      {1801439850948184, 1351079888211148, 450359962737049, 900719925474099, 1801439850948198},
      {1, 0, 0, 0, 0},
      {1841354044333475, 16398895984059, 755974180946558, 900171276175154, 1821297809914039},
  };

  SECTION("doubling gives us the same result as adding an element to itself") {
    c21t::element_p3 res, res2;

    add(res, g, g);

    c21t::element_p1p1 res2_p1p1;
    double_element(res2_p1p1, g);
    c21t::to_element_p3(res2, res2_p1p1);

    REQUIRE(res == res2);
  }
}
