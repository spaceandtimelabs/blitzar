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
#include "sxt/execution/async/continuation_fn.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::xena;

TEST_CASE("we can determine if a given functor can serve as a continuation") {
  SECTION("we handle functors that aren't noexcept") {
    auto f1 = []() {};
    REQUIRE(!continuation_fn<decltype(f1), int, int>);
    REQUIRE(!continuation_fn<decltype(f1), void, void>);
  }

  SECTION("we handle void functors") {
    auto f1 = []() noexcept {};
    REQUIRE(continuation_fn<decltype(f1), void, void>);

    auto f2 = []() noexcept { return 123; };
    REQUIRE(continuation_fn<decltype(f2), void, int>);
  }

  SECTION("we handle non-void functors") {
    auto f1 = [](int x) noexcept { return x; };
    REQUIRE(continuation_fn<decltype(f1), int, int>);
  }

  SECTION("we handle convertible functors") {
    auto f1 = [](int x) noexcept { return x; };
    REQUIRE(continuation_fn<decltype(f1), int, long>);
  }
}
