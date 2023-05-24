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
#include "sxt/base/container/span_iterator.h"

#include <vector>

#include "sxt/base/test/unit_test.h"

using namespace sxt::basct;

TEST_CASE("we can iterator over spans of data") {
  SECTION("iterate over clumps of 3") {
    std::vector<int> data = {1, 2, 3, 4, 5, 6};
    span_iterator<int> it{data.data(), 3};
    span_iterator<int> last{data.data() + data.size(), 3};
    std::vector<span<int>> v(it, last);
    REQUIRE(v.size() == 2);
    REQUIRE(v[0].data() == &data[0]);
    REQUIRE(v[0].size() == 3);
    REQUIRE(v[1].data() == &data[3]);
    REQUIRE(v[1].size() == 3);
  }
}
