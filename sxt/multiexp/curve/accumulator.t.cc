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
#include "sxt/multiexp/curve/accumulator.h"

#include "sxt/algorithm/base/reducer.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/double.h"
#include "sxt/curve21/operation/neg.h"
#include "sxt/curve21/operation/overload.h"
#include "sxt/curve21/type/element_p3.h"

using namespace sxt;
using namespace sxt::mtxcrv;

TEST_CASE("accumulator reduces curve21 elements") {
  c21t::element_p3 e1{
      {3990542415680775, 3398198340507945, 4322667446711068, 2814063955482877, 2839572215813860},
      {1801439850948184, 1351079888211148, 450359962737049, 900719925474099, 1801439850948198},
      {1, 0, 0, 0, 0},
      {1841354044333475, 16398895984059, 755974180946558, 900171276175154, 1821297809914039},
  };

  REQUIRE(algb::reducer<accumulator<c21t::element_p3>>);

  SECTION("we can accumulate curve21 elements") {
    auto p1 = 2 * e1;
    auto p2 = 3 * e1;
    accumulator<c21t::element_p3>::accumulate_inplace(p1, p2);
    REQUIRE(p1 == 5 * e1);
  }
}
