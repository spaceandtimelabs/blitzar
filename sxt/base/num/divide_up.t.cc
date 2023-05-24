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
#include "sxt/base/num/divide_up.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt::basn;

TEST_CASE("we can perform division rounded up") {
  REQUIRE(divide_up(0, 3) == 0);
  REQUIRE(divide_up(1, 3) == 1);
  REQUIRE(divide_up(2, 3) == 1);
  REQUIRE(divide_up(3, 3) == 1);
  REQUIRE(divide_up(3, 3) == 1);
  REQUIRE(divide_up(4, 3) == 2);
}
