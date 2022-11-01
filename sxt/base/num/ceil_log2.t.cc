#include "sxt/base/num/ceil_log2.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt::basn;

TEST_CASE("we find the ceil log2 of a number") {
  REQUIRE(ceil_log2(1) == 0);
  REQUIRE(ceil_log2(2) == 1);
  REQUIRE(ceil_log2(3) == 2);
  REQUIRE(ceil_log2(4) == 2);
  REQUIRE(ceil_log2(5) == 3);
  REQUIRE(ceil_log2(6) == 3);
  REQUIRE(ceil_log2(7) == 3);
  REQUIRE(ceil_log2(8) == 3);
  REQUIRE(ceil_log2(9) == 4);
  REQUIRE(ceil_log2(1ULL << 63) == 63);
  REQUIRE(ceil_log2((1ULL << 63) + 1) == 64);
  REQUIRE(ceil_log2(0xffffffffffffffff) == 64);
}
