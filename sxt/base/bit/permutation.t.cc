#include "sxt/base/bit/permutation.h"

#include <iostream>

#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::basbt;

TEST_CASE("we can compute the next bit permutation") {
  SECTION("we can permute over a single bit") {
    REQUIRE(next_permutation(0b1u) == 0b10u);
    REQUIRE(next_permutation(0b10u) == 0b100u);
  }

  SECTION("we can permute over two bits") {
    REQUIRE(next_permutation(0b11u) == 0b101u);
    REQUIRE(next_permutation(0b101u) == 0b110u);
    REQUIRE(next_permutation(0b110u) == 0b1001u);
  }
}
