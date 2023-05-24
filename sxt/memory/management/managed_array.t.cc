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
#include "sxt/memory/management/managed_array.h"

#include "sxt/base/test/allocator_aware.h"
#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::memmg;

TEST_CASE("managed_array is an allocator-aware container manages an array of "
          "trivially destructible objects") {
  SECTION("we can construct a managed array from an initializer list") {
    managed_array<int> arr{1, 2, 3};
    REQUIRE(arr.size() == 3);
    REQUIRE(arr[0] == 1);
    REQUIRE(arr[1] == 2);
    REQUIRE(arr[2] == 3);
  }

  SECTION("we can move-assign a managed array to a void array") {
    managed_array<int> arr1{1, 2, 3};
    auto data = arr1.data();
    managed_array<void> arr2 = std::move(arr1);
    REQUIRE(arr1.data() == nullptr);
    REQUIRE(arr2.data() == data);
  }

  SECTION("we can operate on an array through the void interface") {
    managed_array<int> arr1{1, 2, 3};
    managed_array<void>& arr2 = arr1;
    arr2.reset();
    REQUIRE(arr1.empty());
  }

  SECTION("verify allocator aware operations") {
    managed_array<int> arr{1, 2, 3};
    bastst::exercise_allocator_aware_operations(arr);
  }

  SECTION("we can shrink an array") {
    managed_array<int> arr{1, 2, 3};
    arr.shrink(2);
    managed_array<int> expected{1, 2};
    REQUIRE(arr == expected);
  }
}
