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
#include "sxt/base/container/span.h"

#include <vector>

#include "sxt/base/test/unit_test.h"

using namespace sxt::basct;

TEST_CASE("span represents a view into a contiguous region of memory") {
  SECTION("a span is default constructed to be empty") {
    span<int> s;
    REQUIRE(s.size() == 0);
    REQUIRE(s.data() == nullptr);
    REQUIRE(s.empty());
  }

  SECTION("we can use span to access an array") {
    int data[] = {1, 2, 3, 4, 5};
    span<int> s{data, 3};
    REQUIRE(s.data() == data);
    REQUIRE(s.size() == 3);
    REQUIRE(s[0] == 1);
    REQUIRE(s[1] == 2);
    REQUIRE(s[2] == 3);
  }

  SECTION("span is implicitly constructible from containers") {
    std::vector<int> v = {1, 2, 3};
    span<int> s{v};
    REQUIRE(s.data() == v.data());
    REQUIRE(s.size() == v.size());
  }

  SECTION("we can implicitly convert to a void span") {
    std::vector<int> v = {1, 2, 3};

    span<int> s1{v};
    span_void sv1{s1};
    REQUIRE(sv1.data() == v.data());

    cspan<int> s2{v};
    span_cvoid sv2{s2};
    REQUIRE(sv2.data() == v.data());
  }
}
