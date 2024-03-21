#include "sxt/base/bit/permutation.h"

#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::basbt;

TEST_CASE("we can compute the next bit permutation") {
  REQUIRE(next_permutation(0b00010011u) == 0b00010101u);
}
