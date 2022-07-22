#include "sxt/multiexp/base/exponent_utility.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/multiexp/base/exponent.h"

using namespace sxt::mtxb;

TEST_CASE("we can or the bits of exponents") {
  exponent e;

  SECTION("or-ing zero does nothing") {
    uint8_t zeros[4] = {};
    or_equal(e, zeros);
    REQUIRE(e == exponent{});
  }

  SECTION("we can or non-zero values") {
    uint8_t val[4] = {0, 1};
    or_equal(e, val);
    REQUIRE(e == exponent{256, 0, 0, 0});
  }
}

TEST_CASE("we can count the number of nonzero digits") {
  SECTION("we correctly count when there are no nonzero digits") {
    exponent e;
    REQUIRE(count_nonzero_digits(e, 3) == 0);
  }

  SECTION("we correctly count non-zero digits when present") {
    exponent e{0b1, 0, 0, 0};
    REQUIRE(count_nonzero_digits(e, 3) == 1);

    e = exponent{0b1010, 0, 0, 0};
    REQUIRE(count_nonzero_digits(e, 3) == 2);

    e = exponent{0b1010, 0, 0, 0b1};
    REQUIRE(count_nonzero_digits(e, 3) == 3);
  }
}
