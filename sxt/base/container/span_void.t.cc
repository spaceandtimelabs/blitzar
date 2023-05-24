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
#include "sxt/base/container/span_void.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt::basct;

TEST_CASE("we track a span of unknown elements") {
  SECTION("we start empty") {
    span_void s;
    REQUIRE(s.empty());
  }

  SECTION("span_void is convertible to span_cvoid") {
    int array[] = {1, 2, 3};
    span_void s{array, 3, sizeof(int)};
    span_cvoid s2 = s;
    REQUIRE(s2.data() == array);
    REQUIRE(s2.size() == 3);
    REQUIRE(s2.element_size() == sizeof(int));
  }
}
