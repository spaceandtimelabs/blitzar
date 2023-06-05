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
#include "sxt/execution/async/coroutine.h"

#include "sxt/base/device/stream.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"

using namespace sxt;
using namespace sxt::xena;

static future<> f_v();
static future<int> f_i();
static future<int> f_i2();

TEST_CASE("futures interoperate with coroutines") {
  SECTION("we can handle a void coroutine") {
    auto res = f_v();
    REQUIRE(res.ready());
  }

  SECTION("we can handle an int coroutine") {
    auto res = f_i();
    REQUIRE(res.ready());
    REQUIRE(res.value() == 123);
  }

  SECTION("we handle a chained coroutine") {
    auto res = f_i2();
    REQUIRE(res.ready());
    REQUIRE(res.value() == 124);
  }
}

static future<> f_v() { co_return; }

static future<int> f_i() { co_return 123; }

static future<int> f_i2() {
  auto x = co_await f_i();
  co_return x + 1;
}
