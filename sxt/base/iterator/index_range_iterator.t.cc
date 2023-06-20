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
#include "sxt/base/iterator/index_range_iterator.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::basit;

TEST_CASE("we can iterate over index_range's") {
  SECTION("we can compare index_range iterators") {
    index_range_iterator it1{index_range{0, 3}, 1};
    REQUIRE(it1 == it1);
    index_range_iterator it2{index_range{1, 3}, 1};
    REQUIRE(it1 != it2);
  }

  SECTION("we can increment an index_range iterator") {
    index_range_iterator it1{index_range{0, 3}, 1}, it2{index_range{1, 3}, 1};
    REQUIRE(it1 != it2);
    ++it1;
    REQUIRE(it1 == it2);
  }

  SECTION("we can iterate an index range with a step") {
    index_range_iterator it1{index_range{0, 3}, 2};
    ++it1;
    REQUIRE(*it1 == index_range{2, 3});
  }

  SECTION("we can compute the distance between iterators") {
    index_range_iterator it1{index_range{0, 3}, 1}, it2{index_range{2, 3}, 1};
    REQUIRE(it2 - it1 == 2);
    REQUIRE(it1 - it2 == -2);
  }

  SECTION("we can compute distance with a non-unity step") {
    index_range_iterator it1{index_range{0, 3}, 2}, it2{index_range{2, 3}, 2};
    REQUIRE(it2 - it1 == 1);
    it1 = {index_range{1, 3}, 2};
    REQUIRE(it2 - it1 == 1);
  }

  SECTION("we can advance an iterator backwards") {
    index_range_iterator it1{index_range{3, 3}, 1};
    it1 -= 2;
    REQUIRE(*it1 == index_range{1, 2});
  }
}
