#include "sxt/multiexp/base/exponent.h"

#include <limits>

#include "sxt/base/test/unit_test.h"
using namespace sxt::mtxb;

TEST_CASE("we can determine the position of the highest 1 bit") {
  constexpr auto u64_max = std::numeric_limits<uint64_t>::max();

  SECTION("we return -1 if no bit is set") {
    exponent e;
    REQUIRE(e.highest_bit() == -1);
  }

  SECTION("if the exponent has a 1 bit, we return the highest 1-bit index") {
    exponent e{0b1, 0, 0, 0};
    REQUIRE(e.highest_bit() == 0);

    e = exponent{0b101, 0, 0, 0};
    REQUIRE(e.highest_bit() == 2);

    e = exponent{0b101, 0b1, 0, 0};
    REQUIRE(e.highest_bit() == 64);

    e = exponent{0b101, u64_max, 0, 0};
    REQUIRE(e.highest_bit() == 127);

    e = exponent{u64_max, u64_max, u64_max, u64_max};
    REQUIRE(e.highest_bit() == 255);
  }
}

TEST_CASE("we can compare exponents") {
  exponent e1{1, 0, 0, 0};
  exponent e2{0, 1, 0, 0};
  REQUIRE(e1 < e2);
}
