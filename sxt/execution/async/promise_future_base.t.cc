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
#include "sxt/execution/async/promise_future_base.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::xena;

TEST_CASE("future_base and promise_base track each other") {
  SECTION("we can default construct") {
    future_base fut;
    REQUIRE(fut.promise() == nullptr);

    promise_base pr;
    REQUIRE(pr.future() == nullptr);
  }

  SECTION("pointers are updated on move construction") {
    future_base fut;
    promise_base pr;

    fut.set_promise(&pr);
    pr.set_future(&fut);

    promise_base pr_p{std::move(pr)};
    REQUIRE(fut.promise() == &pr_p);
    REQUIRE(pr.future() == nullptr);
    REQUIRE(pr_p.future() == &fut);

    future_base fut_p{std::move(fut)};
    REQUIRE(fut.promise() == nullptr);
    REQUIRE(pr_p.future() == &fut_p);
    REQUIRE(fut_p.promise() == &pr_p);
  }

  SECTION("pointers are updated on move assigned") {
    future_base fut;
    promise_base pr;

    fut.set_promise(&pr);
    pr.set_future(&fut);

    promise_base pr_p;
    pr_p = std::move(pr);
    REQUIRE(fut.promise() == &pr_p);
    REQUIRE(pr.future() == nullptr);
    REQUIRE(pr_p.future() == &fut);

    future_base fut_p;
    fut_p = std::move(fut);
    REQUIRE(fut.promise() == nullptr);
    REQUIRE(pr_p.future() == &fut_p);
    REQUIRE(fut_p.promise() == &pr_p);
  }
}
