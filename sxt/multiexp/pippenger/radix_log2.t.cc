#include "sxt/multiexp/pippenger/radix_log2.h"

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::mtxpi;

TEST_CASE("we can compute the radix of arbitrary exponents") {
  SECTION("we correctly handle zero inputs and zero outputs") {
    uint8_t max_exponent[] = {1, 2, 3};

    REQUIRE(compute_radix_log2(max_exponent, 0, 1) == 1);
    REQUIRE(compute_radix_log2(max_exponent, 1, 0) == 1);
    REQUIRE(compute_radix_log2(max_exponent, 0, 0) == 1);
  }

  SECTION("we handle a max_exponent of zero") {
    uint8_t max_exponent[] = {0};

    REQUIRE(compute_radix_log2(max_exponent, 1, 1) == 1);
  }

  SECTION("we correctly handle different input and output values") {
    uint8_t max_exponent[] = {1, 2, 3};
    REQUIRE(compute_radix_log2(max_exponent, 2, 1) == 18);
    REQUIRE(compute_radix_log2(max_exponent, 1, 2) == 3);
  }
}
