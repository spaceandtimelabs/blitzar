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
#include "sxt/base/type/literal.h"

#include <limits>

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::bast;

TEST_CASE("we can parse literals into arrays of 64-bit integers") {
  SECTION("we can parse a single literal") {
    std::array<uint64_t, 1> vals = {123};
    parse_literal<1, '0', 'x', '2'>(vals);
    std::array<uint64_t, 1> expected = {2};
    REQUIRE(vals == expected);
  }

  SECTION("we can handle multiple digits") {
    std::array<uint64_t, 1> vals = {123};
    parse_literal<1, '0', 'x', '1', '0'>(vals);
    std::array<uint64_t, 1> expected = {16};
    REQUIRE(vals == expected);
  }

  SECTION("we handle the maximum number") {
    std::array<uint64_t, 1> vals = {123};
    parse_literal<1, '0', 'x', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
                  'F', 'F'>(vals);
    std::array<uint64_t, 1> expected = {std::numeric_limits<uint64_t>::max()};
    REQUIRE(vals == expected);
  }

  SECTION("we handle multiple int64s") {
    std::array<uint64_t, 2> vals = {123};
    parse_literal<2, '0', 'x', '3', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F',
                  'F', 'F', 'F'>(vals);
    std::array<uint64_t, 2> expected = {std::numeric_limits<uint64_t>::max(), 3};
    REQUIRE(vals == expected);
  }
}
