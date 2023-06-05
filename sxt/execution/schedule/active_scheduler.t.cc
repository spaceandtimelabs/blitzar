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
#include "sxt/execution/schedule/active_scheduler.h"

#include <iostream>
#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/test_pollable_event.h"

using namespace sxt;
using namespace sxt::xens;

TEST_CASE("we can schedule pollable events to run") {
  active_scheduler sched;

  std::vector<int> ids;
  auto f = [&](int id) noexcept { ids.push_back(id); };

  SECTION("we can run with no events") { sched.run(); }

  SECTION("we can schedule a single event") {
    sched.schedule(std::make_unique<test_pollable_event>(123, 1, f));
    sched.run([&](int device) { REQUIRE(device == 123); });
    std::vector<int> expected = {123};
    REQUIRE(ids == expected);
  }

  SECTION("we can schedule multiple events") {
    sched.schedule(std::make_unique<test_pollable_event>(1, 10, f));
    sched.schedule(std::make_unique<test_pollable_event>(2, 5, f));
    sched.run();
    std::vector<int> expected = {2, 1};
    REQUIRE(ids == expected);
  }
}
