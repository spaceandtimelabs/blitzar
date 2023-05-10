/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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
#include "sxt/algorithm/base/gather_mapper.h"

#include "sxt/algorithm/base/mapper.h"
#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::algb;

TEST_CASE("we can map contiguous indexes to gather reads") {
  SECTION("gather_mapper satisfies the mapper concept") { REQUIRE(mapper<gather_mapper<int>>); }

  SECTION("we can index a block of remapped data") {
    int data[] = {1, 2, 3, 4};
    unsigned indexes[] = {0, 2, 1, 3};
    gather_mapper<int> mapper{data, indexes};

    REQUIRE(mapper.map_index(0) == 1);
    REQUIRE(mapper.map_index(1) == 3);
    REQUIRE(mapper.map_index(2) == 2);
    int x;
    mapper.map_index(x, 1);
    REQUIRE(x == 3);
  }
}
