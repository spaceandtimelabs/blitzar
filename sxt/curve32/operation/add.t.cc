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
      {0x325d51a, 0x18b5823, 0xf6592a, 0x104a92d, 0x1a4b31d, 0x1d6dc5c, 0x27118fe, 0x7fd814,
       0x13cd6e5, 0x85a4db},
      {0x2666658, 0x1999999, 0xcccccc, 0x1333333, 0x1999999, 0x666666, 0x3333333, 0xcccccc,
       0x2666666, 0x1999999},
      {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0x1b7dda3, 0x1a2ace9, 0x25eadbb, 0x3ba8a, 0x83c27e, 0xabe37d, 0x1274732, 0xccacdd, 0xfd78b7,
       0x19e1d7c},
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
