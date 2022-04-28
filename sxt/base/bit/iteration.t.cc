#include "sxt/base/bit/iteration.h"

#include "sxt/base/test/unit_test.h"
using namespace sxt::basbt;

TEST_CASE("we can iterate through the bits that are set in an integer") {
  uint64_t bitset;

  SECTION("we can iterate over a number with only a single bit set") {
    bitset = 1;
    REQUIRE(consume_next_bit(bitset) == 0);
    REQUIRE(bitset == 0);

    bitset = 1 << 1;
    REQUIRE(consume_next_bit(bitset) == 1);
    REQUIRE(bitset == 0);

    bitset = 1 << 2;
    REQUIRE(consume_next_bit(bitset) == 2);
    REQUIRE(bitset == 0);
  }

  SECTION("we can iterate through a number with two bits set") {
    bitset = 1 | (1 << 1);
    REQUIRE(consume_next_bit(bitset) == 0);
    REQUIRE(bitset != 0);
    REQUIRE(consume_next_bit(bitset) == 1);
    REQUIRE(bitset == 0);

    bitset = 1 | (1 << 2);
    REQUIRE(consume_next_bit(bitset) == 0);
    REQUIRE(bitset != 0);
    REQUIRE(consume_next_bit(bitset) == 2);
    REQUIRE(bitset == 0);

    bitset = (1 << 3) | (1 << 7);
    REQUIRE(consume_next_bit(bitset) == 3);
    REQUIRE(bitset != 0);
    REQUIRE(consume_next_bit(bitset) == 7);
    REQUIRE(bitset == 0);
  }
}
