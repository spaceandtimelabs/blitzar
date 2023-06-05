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
#include "sxt/execution/schedule/pending_scheduler.h"

#include <tuple>
#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/test_pending_event.h"

using namespace sxt;
using namespace sxt::xens;

TEST_CASE("we can schedule pending events that will run when resources are available") {
  int num_devices = 3;
  size_t target_max_active = 2;
  pending_scheduler scheduler{static_cast<size_t>(num_devices), target_max_active};

  std::vector<std::tuple<int, int>> ids;
  auto f = [&](int id, int device) noexcept {
    ids.emplace_back(id, device);
    scheduler.on_event_new(device);
  };

  SECTION("we can construct a pending scheduler with no devices") {
    pending_scheduler s{0, 5};
    REQUIRE(s.get_available_device() == -1);
  }

  SECTION("we can choose the first available device with the lowest usage") {
    REQUIRE(scheduler.get_available_device() == 0);
    scheduler.on_event_new(0);
    REQUIRE(scheduler.get_available_device() == 1);
    scheduler.on_event_new(1);
    scheduler.on_event_new(2);
    REQUIRE(scheduler.get_available_device() == 0);
    scheduler.on_event_new(0);
    REQUIRE(scheduler.get_available_device() == 1);
  }

  SECTION("we handle the case of no devices available") {
    for (int device = 0; device < num_devices; ++device) {
      for (size_t i = 0; i < target_max_active; ++i) {
        scheduler.on_event_new(device);
      }
    }
    REQUIRE(scheduler.get_available_device() == -1);
  }

  SECTION("we invoke events as resources become available") {
    for (int device = 0; device < num_devices; ++device) {
      for (size_t i = 0; i < target_max_active; ++i) {
        scheduler.on_event_new(device);
      }
    }
    scheduler.schedule(std::make_unique<test_pending_event>(1, f));
    scheduler.on_event_done(2);
    std::vector<std::tuple<int, int>> expected = {{1, 2}};
    REQUIRE(ids == expected);
  }

  SECTION("if pending_events don't start new events, we may invoke multiple events as a device "
          "becomes available") {
    for (int device = 0; device < num_devices; ++device) {
      for (size_t i = 0; i < target_max_active; ++i) {
        scheduler.on_event_new(device);
      }
    }
    auto fp = [&](int id, int device) noexcept { ids.emplace_back(id, device); };
    scheduler.schedule(std::make_unique<test_pending_event>(10, fp));
    scheduler.schedule(std::make_unique<test_pending_event>(20, fp));
    scheduler.on_event_done(2);
    std::vector<std::tuple<int, int>> expected = {{20, 2}, {10, 2}};
    REQUIRE(ids == expected);
  }

  SECTION("we can schedule multiple events") {
    for (int device = 0; device < num_devices; ++device) {
      for (size_t i = 0; i < target_max_active; ++i) {
        scheduler.on_event_new(device);
      }
    }
    scheduler.schedule(std::make_unique<test_pending_event>(1, f));
    scheduler.schedule(std::make_unique<test_pending_event>(2, f));
    scheduler.on_event_done(2);
    scheduler.on_event_done(0);
    std::vector<std::tuple<int, int>> expected = {{2, 2}, {1, 0}};
    REQUIRE(ids == expected);
  }
}
