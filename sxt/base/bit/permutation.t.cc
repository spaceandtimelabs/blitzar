#include "sxt/base/bit/permutation.h"

#include <iostream>

#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::basbt;

TEST_CASE("we can compute the next bit permutation") {
  REQUIRE(next_permutation(0b00010011u) == 0b00010101u);
  std::cout << 0b11011011101111 << "\n";
  std::cout << next_permutation(0b11011011011111) << "\n";
  std::cout << 0b11011011011111 << "\n";
  std::cout << 0b11111111111 << "\n";
  // 11011011101111
}
