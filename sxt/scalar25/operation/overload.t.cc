#include "sxt/scalar25/operation/overload.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::s25t;

TEST_CASE("we can use operators on scalars") {
  auto x = 0x1_s25;

  SECTION("we can use basic operations") {
    REQUIRE(0x1_s25 + 0x2_s25 == 0x3_s25);
    REQUIRE(0x2_s25 - 0x1_s25 == 0x1_s25);
    REQUIRE(0x2_s25 * 0x3_s25 == 0x6_s25);
    REQUIRE(-0x2_s25 == 0x0_s25 - 0x2_s25);
    REQUIRE(0x6_s25 / 0x2_s25 == 0x3_s25);
  }

  SECTION("we can use +=") {
    x += 0x2_s25;
    REQUIRE(x == 0x3_s25);
  }

  SECTION("we can use -=") {
    x -= 0x2_s25;
    REQUIRE(x == -0x1_s25);
  }

  SECTION("we can use *=") {
    x *= 0x2_s25;
    REQUIRE(x == 0x2_s25);
  }

  SECTION("we can use /=") {
    x /= 0x2_s25;
    REQUIRE(x * 0x2_s25 == 0x1_s25);
  }
}
