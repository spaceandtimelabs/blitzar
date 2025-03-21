#include "sxt/execution/async/shared_future_state.h"

#include "sxt/base/test/unit_test.h"

TEST_CASE("todo") {}
// shared future is kept alive even if there are no references to it
// shared future is destroyed
// we can create a future from a ready state
// we can create a future from shared future state
// we can create multiple futures from shared future state
// we can create both events with a value and void events
