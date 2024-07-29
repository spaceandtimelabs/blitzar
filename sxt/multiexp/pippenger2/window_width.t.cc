#include "sxt/multiexp/pippenger2/window_width.h"

#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can get a default window width for Pippengers partition algorithm") {
  auto width = get_default_window_width();
  REQUIRE(width > 0);
}
