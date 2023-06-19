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

#include "sxt/base/device/active_device_guard.h"
#include "sxt/base/device/property.h"
#include "sxt/base/device/state.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"

using namespace sxt;
using namespace sxt::xena;

static future<> f_v() noexcept;
static future<int> f_i() noexcept;
static future<int> f_i2() noexcept;
static future<int> f_dev(promise<int>& p);

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

  SECTION("coroutines restore the active device") {
    basdv::active_device_guard active_guard{0};
    promise<int> p;
    auto res = f_dev(p);
    REQUIRE(!res.ready());
    p.set_value(123);
    REQUIRE(res.value() == 124);
    REQUIRE(basdv::get_device() == 0);
  }
}

static future<> f_v() noexcept { co_return; }

static future<int> f_i() noexcept { co_return 123; }

static future<int> f_i2() noexcept {
  auto x = co_await f_i();
  co_return x + 1;
}

static future<int> f_dev(promise<int>& p) {
  auto device = basdv::get_num_devices() - 1;
  basdv::active_device_guard active_guard{device};
  auto val = co_await future<int>{p} + 1;
  REQUIRE(basdv::get_device() == device);
  co_return val;
}
