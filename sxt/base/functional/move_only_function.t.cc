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
#include "sxt/base/functional/move_only_function.h"

#include <memory>

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::basf;

TEST_CASE("move_only_function is a type-erased container for a move-only functor") {
  SECTION("we can default construct move_only_function") {
    move_only_function<int()> f;
    REQUIRE(!f);
  }

  SECTION("we can use a lambda with move_only_function") {
    move_only_function<int()> f{[]() { return 123; }};
    REQUIRE(!noexcept(f()));
    REQUIRE(f);
    REQUIRE(f() == 123);
  }

  SECTION("we can use move_only_function with functors that take arguments") {
    move_only_function<int(int)> f{[](int x) { return x; }};
    REQUIRE(f);
    REQUIRE(f(123) == 123);
  }

  SECTION("we make noexcept functors") {
    move_only_function<int() noexcept> f{[]() { return 123; }};
    REQUIRE(noexcept(f()));
    REQUIRE(f);
    REQUIRE(f() == 123);
  }

  SECTION("we can use functors that would not be copy constructible") {
    auto ptr = std::make_unique<int>(123);
    auto f = [x = std::move(ptr)] { return *x; };
    move_only_function<int()> ff{std::move(f)};
    REQUIRE(ff() == 123);
  }
}
