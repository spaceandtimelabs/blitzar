#include "sxt/multiexp/base/digit_utility.h"

#include <array>

#include "sxt/base/test/unit_test.h"

using namespace sxt::mtxb;

TEST_CASE("we can extract digits given a radix that's a power of two") {
  SECTION("we can extract digits from a byte using a radix of 2^8") {
    uint8_t val[] = {0b10};
    REQUIRE(extract_digit(val, 8, 0) == 0b10);
  }

  SECTION("we can extract digits from a byte with a radix less than 2^8") {
    uint8_t val[] = {0b10110011};

    REQUIRE(extract_digit(val, 3, 0) == 0b011);
    REQUIRE(extract_digit(val, 3, 1) == 0b110);
    REQUIRE(extract_digit(val, 3, 2) == 0b010);
  }

  SECTION("we can extract digits from a multi-byte number") {
    uint8_t val[] = {0b10110011, 0b11001011};

    REQUIRE(extract_digit(val, 8, 0) == val[0]);
    REQUIRE(extract_digit(val, 8, 1) == val[1]);

    REQUIRE(extract_digit(val, 3, 0) == 0b011);
    REQUIRE(extract_digit(val, 3, 1) == 0b110);
    REQUIRE(extract_digit(val, 3, 2) == 0b110);
    REQUIRE(extract_digit(val, 3, 3) == 0b101);
    REQUIRE(extract_digit(val, 3, 4) == 0b100);
    REQUIRE(extract_digit(val, 3, 5) == 0b001);
  }
}

TEST_CASE("we can count the number of non-zero digits in an exponent") {
  SECTION("we correctly count exponents with the highest bit at position 0") {
    uint8_t val[] = {1, 0};
    REQUIRE(count_nonzero_digits(val, 0, 3) == 1);
  }

  SECTION("we correctly count exponents with the highest bit at the end") {
    uint8_t val[] = {1, 0b10000000};
    REQUIRE(count_nonzero_digits(val, 15, 3) == 2);
  }
}

TEST_CASE("we can count the number of digits in a number") {
  SECTION("we correctly count zero digits") {
    REQUIRE(count_num_digits(std::array<uint8_t, 1>{0}, 2) == 0);
  }

  SECTION("we correctly count a single digit") {
    REQUIRE(count_num_digits(std::array<uint8_t, 1>{1}, 2) == 1);
  }

  SECTION("we correctly count the maximum number of digits") {
    REQUIRE(count_num_digits(std::array<uint8_t, 1>{255}, 2) == 4);
  }
}
