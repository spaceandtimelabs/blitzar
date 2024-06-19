#include "sxt/base/num/round_up.h"

#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::basn;

TEST_CASE("we can round a number up to the nearest multiple") {
  int x = 4;
  int y;

  SECTION("if x is already a multiple of y, we return the identity") {
    y = 2;
    REQUIRE(round_up(x, y) == x);
  }

  SECTION("if x is not already a multiple, we return the next multiple") {
    y = 3;
    REQUIRE(round_up(x, y) == 6);
  }
}
