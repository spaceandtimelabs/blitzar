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
#include "sxt/base/container/blob_array.h"

#include <iterator>

#include "sxt/base/test/unit_test.h"

using namespace sxt::basct;

TEST_CASE("we can manage an array of blobs") {
  blob_array arr;

  SECTION("an array starts empty") {
    REQUIRE(arr.empty());
    REQUIRE(arr.size() == 0);
  }

  SECTION("we can construct an array with elements") {
    arr = blob_array{10, 2};
    REQUIRE(arr.size() == 10);
    REQUIRE(arr.blob_size() == 2);
    REQUIRE(arr[0].size() == 2);
    REQUIRE(std::distance(arr[0].data(), arr[1].data()) == 2);
  }
}
