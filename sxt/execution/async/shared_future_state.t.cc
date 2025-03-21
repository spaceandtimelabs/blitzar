#include "sxt/execution/async/shared_future_state.h"

#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::xena;

TEST_CASE("we can manage shared future state") {
  promise<int> ps;
  auto s = std::make_shared<shared_future_state<int>>(future<int>{ps});

  SECTION("we can create a future from a shared ready state") {
    ps.set_value(123);
  }
}
// shared future is kept alive even if there are no references to it
// shared future is destroyed
// we can create a future from a ready state
// we can create a future from shared future state
// we can create multiple futures from shared future state
// we can create both events with a value and void events
