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
#include "sxt/base/num/ceil_log2.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt::basn;

TEST_CASE("we find the ceil log2 of a number") {
  REQUIRE(ceil_log2(1) == 0);
  REQUIRE(ceil_log2(2) == 1);
  REQUIRE(ceil_log2(3) == 2);
  REQUIRE(ceil_log2(4) == 2);
  REQUIRE(ceil_log2(5) == 3);
  REQUIRE(ceil_log2(6) == 3);
  REQUIRE(ceil_log2(7) == 3);
  REQUIRE(ceil_log2(8) == 3);
  REQUIRE(ceil_log2(9) == 4);
  REQUIRE(ceil_log2(1ULL << 63) == 63);
  REQUIRE(ceil_log2((1ULL << 63) + 1) == 64);
  REQUIRE(ceil_log2(0xffffffffffffffff) == 64);
}
