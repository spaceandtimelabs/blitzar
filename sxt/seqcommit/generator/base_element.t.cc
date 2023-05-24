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
#include "sxt/seqcommit/generator/base_element.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/type/element_p3.h"

using namespace sxt;
using namespace sxt::sqcgn;

TEST_CASE("we can deterministically generate base elements for a given row index") {
  c21t::element_p3 p1, p2;

  SECTION("we generate different base elements for different indexes") {
    compute_base_element(p1, 0);
    compute_base_element(p2, 1);
    REQUIRE(p1 != p2);
  }

  SECTION("we generate the same base element for the same row index") {
    compute_base_element(p1, 1);
    compute_base_element(p2, 1);
    REQUIRE(p1 == p2);
  }
}
