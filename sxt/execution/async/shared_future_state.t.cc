#include "sxt/execution/async/shared_future_state.h"

#include "sxt/base/test/unit_test.h"
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
// shared future is kept alive even if there are no references to it
// shared future is destroyed
// we can create a future from a ready state
// we can create a future from shared future state
// we can create multiple futures from shared future state
// we can create both events with a value and void events
