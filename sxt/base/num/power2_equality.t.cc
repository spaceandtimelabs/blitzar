#include "sxt/base/num/power2_equality.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt::basn;

TEST_CASE("we check if a number is a power of 2") {
  REQUIRE(!is_power2(0));
  REQUIRE(is_power2(1));
  REQUIRE(is_power2(2));
  REQUIRE(!is_power2(3));
  REQUIRE(is_power2(4));
  REQUIRE(!is_power2(5));
  REQUIRE(!is_power2(6));
  REQUIRE(!is_power2(7));
  REQUIRE(is_power2(8));
  REQUIRE(is_power2(1ULL << 63));
  REQUIRE(!is_power2((1ULL << 63) - 1));
  REQUIRE(!is_power2((1ULL << 63) + 1));
  REQUIRE(!is_power2(0xffffffffffffffff));
  REQUIRE(!is_power2(0xffffffffffffffff - 1));
}
