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
#include "sxt/base/container/stack_array.h"

#include <tuple>

#include "sxt/base/test/unit_test.h"

using namespace sxt::basct;

TEST_CASE("we can construct dynamically-sized arrays on the stack") {
  SXT_STACK_ARRAY(abc, 10, int);
  REQUIRE(abc.size() == 10);
  abc[0] = 123;
  REQUIRE(abc[0] == 123);

  SXT_STACK_ARRAY(xyz, 1 + 2, std::tuple<int, float>);
  REQUIRE(xyz.size() == 3);
}
