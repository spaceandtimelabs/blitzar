#include "sxt/base/num/divide_up.h"

#include "sxt/base/test/unit_test.h"
using namespace sxt::basn;

TEST_CASE("we can perform division rounded up") {
  REQUIRE(divide_up(0, 3) == 0);
  REQUIRE(divide_up(1, 3) == 1);
  REQUIRE(divide_up(2, 3) == 1);
  REQUIRE(divide_up(3, 3) == 1);
  REQUIRE(divide_up(3, 3) == 1);
  REQUIRE(divide_up(4, 3) == 2);
}
