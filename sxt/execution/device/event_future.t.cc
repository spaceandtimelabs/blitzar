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
#include "sxt/execution/device/event_future.h"

#include "sxt/base/device/event_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"

using namespace sxt;
using namespace sxt::xendv;

TEST_CASE("we can manage a future associated with an event") {
  SECTION("event_future is default constructible") {
    event_future<memmg::managed_array<int>> fut;
    REQUIRE(!fut.event());
  }

  SECTION("we can construct an event_future from an event") {
    basdv::event event;
    memmg::managed_array<int> data = {1, 2, 3};
    event_future<memmg::managed_array<int>> fut{std::move(data), 0, std::move(event),
                                                computation_handle{}};
    REQUIRE(fut.event());
  }

  SECTION("events outlive an event_future") {
    basdv::event event;
    auto raw_event = bast::raw_cuda_event_t{event};
    basdv::stream stream;
    {
      memmg::managed_array<int> data = {1, 2, 3};
      event_future<memmg::managed_array<int>> fut{std::move(data), stream.device(),
                                                  std::move(event), computation_handle{}};
    }
    basdv::record_event(raw_event, stream);
  }

  SECTION("we can convert a ready event_future to a future") {
    memmg::managed_array<int> data = {1, 2, 3};
    auto ptr = data.data();
    event_future<memmg::managed_array<int>> fut{std::move(data)};
    REQUIRE(fut.value().data() == ptr);
    xena::future<memmg::managed_array<int>> fut_p{std::move(fut)};
    REQUIRE(fut_p.ready());
    REQUIRE(fut_p.value().data() == ptr);
  }

  SECTION("we can convert a non-ready event_future to a future") {
    basdv::event event;
    basdv::stream stream;
    memmg::managed_array<int> data = {1, 2, 3};
    event_future<memmg::managed_array<int>> fut{std::move(data), stream.device(), std::move(event),
                                                computation_handle{}};
    REQUIRE(fut.event());
    basdv::record_event(*fut.event(), stream);
    xena::future<memmg::managed_array<int>> fut_p{std::move(fut)};
    REQUIRE(!fut_p.ready());
    xens::get_scheduler().run();
    REQUIRE(fut_p.ready());
    memmg::managed_array<int> expected = {1, 2, 3};
    REQUIRE(fut_p.value() == expected);
  }

  SECTION("we can await an event_future") {
    auto fut = []() noexcept -> xena::future<memmg::managed_array<int>> {
      memmg::managed_array<int> data = {1, 2, 3};
      co_return co_await event_future<memmg::managed_array<int>>{std::move(data)};
    }();
    REQUIRE(fut.ready());
    memmg::managed_array<int> expected = {1, 2, 3};
    REQUIRE(fut.value() == expected);
  }
}
