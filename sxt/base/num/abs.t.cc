#include "sxt/base/num/abs.h"

#include "sxt/base/type/int.h"
#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::basn;

TEST_CASE("we can compute the absolute value of numbers") {
  SECTION("we can compute the absolute value of numbers up to 8 bytes") {
    REQUIRE(abs(1) == 1);
    REQUIRE(abs(-1) == 1);
    REQUIRE(abs(-1ll) == 1ll);
  }

  SECTION("we can compute the absolute value of numbers larger than 8 bytes") {
    int128_t x = -1;
    REQUIRE(abs(x) == 1);
  }
}
