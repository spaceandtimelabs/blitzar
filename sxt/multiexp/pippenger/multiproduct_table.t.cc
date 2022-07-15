#include "sxt/multiexp/pippenger/multiproduct_table.h"

#include <vector>

#include "sxt/base/test/unit_test.h"

#include "sxt/multiexp/base/exponent.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/index/index_table.h"

using namespace sxt;
using namespace sxt::mtxpi;

TEST_CASE("we can build an index array from digit bits") {
  std::array<size_t, 8> index_array = {1, 2, 3, 4, 5, 6, 7, 8};

  SECTION("we handle the empty case") {
    make_digit_index_array(index_array, 1, 0);
    std::array<size_t, 8> expected_index_array = {};
    REQUIRE(index_array == expected_index_array);
  }

  SECTION("we handle the case of a single bit") {
    make_digit_index_array(index_array, 1, 0b1);
    std::array<size_t, 8> expected_index_array = {1};
    REQUIRE(index_array == expected_index_array);

    make_digit_index_array(index_array, 1, 0b10);
    expected_index_array = {0, 1};
    REQUIRE(index_array == expected_index_array);
  }

  SECTION("we handle multiple bits") {
    make_digit_index_array(index_array, 1, 0b1010);
    std::array<size_t, 8> expected_index_array = {0, 1, 0, 2};
    REQUIRE(index_array == expected_index_array);
  }
}

TEST_CASE("we can construct a table that identifies the terms needed for "
          "pipppenger's multiproduct phase") {
  mtxi::index_table table;

  SECTION("we correctly handle the empty case with a single input term") {
    std::vector<mtxb::exponent> term_or_all = {mtxb::exponent{}};
    make_multiproduct_term_table(table, {term_or_all.data(), term_or_all.size()}, 3);
    REQUIRE(table == mtxi::index_table{{}});
  }

  SECTION("we correctly handle the empty case with two input terms") {
    std::vector<mtxb::exponent> term_or_all = {mtxb::exponent{}, mtxb::exponent{}};
    make_multiproduct_term_table(table, {term_or_all.data(), term_or_all.size()}, 3);
    REQUIRE(table == mtxi::index_table{{}, {}});
  }

  SECTION("we can construct a table with a single term and a single entry") {
    std::vector<mtxb::exponent> term_or_all = {mtxb::exponent{0b1, 0, 0, 0}};
    make_multiproduct_term_table(table, {term_or_all.data(), term_or_all.size()}, 3);
    REQUIRE(table == mtxi::index_table{{0}});
  }

  SECTION("we can construct a table with a single term and multiple entries") {
    std::vector<mtxb::exponent> term_or_all = {mtxb::exponent{0b1001, 0b1, 0, 0}};
    make_multiproduct_term_table(table, {term_or_all.data(), term_or_all.size()}, 3);
    REQUIRE(table == mtxi::index_table{{0, 1, 21}});
  }

  SECTION("we can construct a table with multiple terms and multiple entries") {
    std::vector<mtxb::exponent> term_or_all = {mtxb::exponent{0b1001, 0, 0, 0},
                                               mtxb::exponent{0b1000, 0, 0, 0}};
    make_multiproduct_term_table(table, {term_or_all.data(), term_or_all.size()}, 3);
    REQUIRE(table == mtxi::index_table{{0, 1}, {1}});
  }
}

TEST_CASE("we can construct a table for the multiproduct computation") {
  mtxi::index_table table;

  SECTION("we handle the empty case") {
    make_multiproduct_table(table, {}, 100, {}, {}, 3);
    REQUIRE(table == mtxi::index_table{});
  }

  SECTION("we handle the case of zero entries") {
    std::vector<uint8_t> exponents = {0};
    std::vector<mtxb::exponent_sequence> sequence = {
        {.element_nbytes = 1, .n = 1, .data = exponents.data()}};
    std::vector<mtxb::exponent> term_or_all = {mtxb::exponent{}};
    std::vector<uint8_t> output_digit_or_all = {0};
    make_multiproduct_table(table, sequence, 100, term_or_all, output_digit_or_all, 3);
    REQUIRE(table == mtxi::index_table{});
  }

  SECTION("we handle the case of a single exponentiation of 1") {
    std::vector<uint8_t> exponents = {1};
    std::vector<mtxb::exponent_sequence> sequence = {
        {.element_nbytes = 1, .n = 1, .data = exponents.data()}};
    std::vector<mtxb::exponent> term_or_all = {mtxb::exponent{1, 0, 0, 0}};
    std::vector<uint8_t> output_digit_or_all = {1};
    make_multiproduct_table(table, sequence, 100, term_or_all, output_digit_or_all, 3);
    REQUIRE(table == mtxi::index_table{{0}});
  }

  SECTION("we handle the case of two exponentiations of 1") {
    std::vector<uint8_t> exponents = {1, 1};
    std::vector<mtxb::exponent_sequence> sequence = {
        {.element_nbytes = 1, .n = exponents.size(), .data = exponents.data()}};
    std::vector<mtxb::exponent> term_or_all = {mtxb::exponent{1, 0, 0, 0},
                                               mtxb::exponent{1, 0, 0, 0}};
    std::vector<uint8_t> output_digit_or_all = {1};
    make_multiproduct_table(table, sequence, 100, term_or_all, output_digit_or_all, 3);
    REQUIRE(table == mtxi::index_table{{0, 1}});
  }

  SECTION("we handle the case of a single exponentiation of 2") {
    std::vector<uint8_t> exponents = {2};
    std::vector<mtxb::exponent_sequence> sequence = {
        {.element_nbytes = 1, .n = 1, .data = exponents.data()}};
    std::vector<mtxb::exponent> term_or_all = {mtxb::exponent{2, 0, 0, 0}};
    std::vector<uint8_t> output_digit_or_all = {2};
    make_multiproduct_table(table, sequence, 100, term_or_all, output_digit_or_all, 3);
    REQUIRE(table == mtxi::index_table{{0}});
  }

  SECTION("we handle the case of a single exponentiation of 3") {
    std::vector<uint8_t> exponents = {3};
    std::vector<mtxb::exponent_sequence> sequence = {
        {.element_nbytes = 1, .n = 1, .data = exponents.data()}};
    std::vector<mtxb::exponent> term_or_all = {mtxb::exponent{3, 0, 0, 0}};
    std::vector<uint8_t> output_digit_or_all = {3};
    make_multiproduct_table(table, sequence, 100, term_or_all, output_digit_or_all, 3);
    REQUIRE(table == mtxi::index_table{{0}, {0}});
  }

  SECTION("we handle the case of two exponentiations of 1 and 2") {
    std::vector<uint8_t> exponents = {1, 2};
    std::vector<mtxb::exponent_sequence> sequence = {
        {.element_nbytes = 1, .n = exponents.size(), .data = exponents.data()}};
    std::vector<mtxb::exponent> term_or_all = {mtxb::exponent{1, 0, 0, 0},
                                               mtxb::exponent{2, 0, 0, 0}};
    std::vector<uint8_t> output_digit_or_all = {0b11};
    make_multiproduct_table(table, sequence, 100, term_or_all, output_digit_or_all, 3);
    REQUIRE(table == mtxi::index_table{{0}, {1}});
  }

  SECTION("we handle the case of two digits") {
    std::vector<uint8_t> exponents = {3};
    std::vector<mtxb::exponent_sequence> sequence = {
        {.element_nbytes = 1, .n = 1, .data = exponents.data()}};
    std::vector<mtxb::exponent> term_or_all = {mtxb::exponent{3, 0, 0, 0}};
    std::vector<uint8_t> output_digit_or_all = {1};
    make_multiproduct_table(table, sequence, 100, term_or_all, output_digit_or_all, 1);
    REQUIRE(table == mtxi::index_table{{0, 1}});
  }
}
