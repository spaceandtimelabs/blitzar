#include "sxt/execution/schedule/scheduler.h"

#include <iostream>
#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/execution/schedule/test_pollable_event.h"

using namespace sxt;
using namespace sxt::xens;

TEST_CASE("we can schedule pollable events to run") {
  scheduler sched;

  std::vector<int> ids;
  auto f = [&](int id) noexcept { ids.push_back(id); };

  SECTION("we can run with no events") { sched.run(); }

  SECTION("we can schedule a single event") {
    sched.schedule(std::make_unique<test_pollable_event>(123, 1, f));
    sched.run();
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
