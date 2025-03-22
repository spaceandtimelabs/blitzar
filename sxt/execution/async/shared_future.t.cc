#include "sxt/execution/async/shared_future.h"

#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::xena;

TEST_CASE("we can manage a shared future") {
  promise<int> ps;
  shared_future<int> fut{future<int>{ps}};
  auto futp = fut.make_future();
  REQUIRE(!futp.ready());
  ps.set_value(123);
  REQUIRE(futp.ready());
  REQUIRE(futp.value() == 123);
}
