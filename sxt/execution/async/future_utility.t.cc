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
#include "sxt/execution/async/future_utility.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/future.h"

using namespace sxt;
using namespace sxt::xena;

static future<> f(promise<>& p1, promise<>& p2) noexcept;

TEST_CASE("we can await multiple futures") {
  promise<> p1, p2;
  auto fut = f(p1, p2);
  REQUIRE(!fut.ready());
  p1.make_ready();
  REQUIRE(!fut.ready());
  p2.make_ready();
  REQUIRE(fut.ready());
}

static future<> f(promise<>& p1, promise<>& p2) noexcept {
  future<> fut1{p1}, fut2{p2};
  return await_all(std::move(fut1), std::move(fut2));
}
