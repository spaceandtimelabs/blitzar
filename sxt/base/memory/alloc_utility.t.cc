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
#include "sxt/base/memory/alloc_utility.h"

#include <memory_resource>
#include <type_traits>

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::basm;

TEST_CASE("we can allocate PODs") {
  std::pmr::monotonic_buffer_resource alloc;

  SECTION("we can allocate a single value") {
    auto obj = allocate_object<double>(&alloc);
    REQUIRE(std::is_same_v<decltype(obj), double*>);
    *obj = 123.456;
  }

  SECTION("we can allocate an array") {
    auto data = allocate_array<double>(&alloc, 3);
    REQUIRE(std::is_same_v<decltype(data), double*>);
    data[0] = 1.23;
    data[1] = 4.56;
    data[2] = 5.78;
  }
}
