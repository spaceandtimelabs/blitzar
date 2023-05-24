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
#include "sxt/algorithm/base/identity_mapper.h"

#include "sxt/algorithm/base/mapper.h"
#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::algb;

TEST_CASE("we can map a contiguous block of data") {
  SECTION("identity_mapper satisfies the mapper concept") { REQUIRE(mapper<identity_mapper<int>>); }

  SECTION("we can index a block of data") {
    int data[] = {1, 2, 3, 4};
    identity_mapper<int> mapper{data};

    REQUIRE(mapper.map_index(0) == 1);
    int x;
    mapper.map_index(x, 1);
    REQUIRE(x == 2);
  }
}
