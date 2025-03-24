/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2025-present Space and Time Labs, Inc.
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
#include "sxt/execution/async/shared_future_state.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/memory/resource/counting_resource.h"

using namespace sxt;
using namespace sxt::xena;

TEST_CASE("we can manage shared future state") {
  promise<int> ps;
  auto s = std::make_shared<shared_future_state<int>>(future<int>{ps});

  SECTION("we can create a future from a shared ready state") {
    ps.set_value(123);
    auto fut = s->make_future();
    REQUIRE(fut.ready());
    REQUIRE(fut.value() == 123);
  }

  SECTION("we can create a future from a shared pending event") {
    auto fut = s->make_future();
    REQUIRE(!fut.ready());
    ps.set_value(123);
    REQUIRE(fut.ready());
    REQUIRE(fut.value() == 123);
  }

  SECTION("we can create multiple futures from a shared state") {
    auto fut1 = s->make_future();
    REQUIRE(!fut1.ready());

    auto fut2 = s->make_future();
    REQUIRE(!fut2.ready());

    ps.set_value(123);

    REQUIRE(fut1.ready());
    REQUIRE(fut1.value() == 123);

    REQUIRE(fut2.ready());
    REQUIRE(fut2.value() == 123);
  }
}

TEST_CASE("the lifetime of future states are properly managed") {
  memr::counting_resource resource;
  REQUIRE(resource.bytes_allocated() == 0);
  promise<int> ps;
  auto s = std::allocate_shared<shared_future_state<int>>(
      std::pmr::polymorphic_allocator<>{&resource}, future<int>{ps});
  REQUIRE(resource.bytes_allocated() > 0);

  SECTION("shared future states are kept alive if there is a pending promise") {
    auto fut = s->make_future();
    s.reset();
    REQUIRE(resource.bytes_deallocated() == 0);
    REQUIRE(s == nullptr);
    REQUIRE(!fut.ready());
    ps.set_value(123);
    REQUIRE(fut.ready());
    REQUIRE(fut.value() == 123);
    REQUIRE(resource.bytes_deallocated() == resource.bytes_allocated());
  }
}

TEST_CASE("we can manage void future states") {
  promise<> ps;
  auto s = std::make_shared<shared_future_state<void>>(future<void>{ps});

  SECTION("we can create a future from a shared ready state") {
    auto fut = s->make_future();
    REQUIRE(!fut.ready());
    ps.make_ready();
    REQUIRE(fut.ready());
  }
}
