/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
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
#include "sxt/multiexp/pippenger2/partition_product.h"

#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can compute the index used to lookup the precomputed sum for a partition") {
  uint8_t scalars[32] = {};

  SECTION("we handle the zero case") {
    auto index = compute_partition_index(scalars, 1, 16, 0);
    REQUIRE(index == 0);
  }

  SECTION("we handle non-zero cases") {
    scalars[0] = 1;
    scalars[2] = 1;
    auto index = compute_partition_index(scalars, 1, 16, 0);
    REQUIRE(index == 5);
  }

  SECTION("we handle a bit index of 2") {
    scalars[0] = 2;
    auto index = compute_partition_index(scalars, 1, 16, 1);
    REQUIRE(index == 1);
  }

  SECTION("we handle n < 16") {
    scalars[0] = 1u;
    scalars[2] = 1u;
    auto index = compute_partition_index(scalars, 1, 2, 0);
    REQUIRE(index == 1);
  }

  SECTION("we handle a step size of 2") {
    scalars[16] = 1;
    auto index = compute_partition_index(scalars, 2, 16, 0);
    REQUIRE(index == 1u << 8u);
  }
}
