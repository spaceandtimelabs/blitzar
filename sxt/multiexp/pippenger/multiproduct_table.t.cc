#include "sxt/multiexp/pippenger/multiproduct_table.h"

#include <vector>

#include "sxt/base/container/blob_array.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/index/index_table.h"

using namespace sxt;
using namespace sxt::mtxpi;

TEST_CASE("we can build an index array from digit bits") {
  std::array<size_t, 16> index_array = {};

  SECTION("we handle the empty case") {
    uint8_t blob[] = {0};
    make_digit_index_array(index_array, 1, blob);
    std::array<size_t, 16> expected_index_array = {};
    REQUIRE(index_array == expected_index_array);
  }

  SECTION("we handle the case of a single bit") {
    uint8_t blob[] = {0b1};
    make_digit_index_array(index_array, 1, blob);
    std::array<size_t, 16> expected_index_array = {1};
    REQUIRE(index_array == expected_index_array);

    blob[0] = 0b10;
    index_array[0] = 99;
    make_digit_index_array(index_array, 1, blob);
    expected_index_array = {99, 1};
    REQUIRE(index_array == expected_index_array);
  }

  SECTION("we handle multiple bits") {
    uint8_t blob[] = {0b1010};
    make_digit_index_array(index_array, 1, blob);
    std::array<size_t, 16> expected_index_array = {0, 1, 0, 2};
    REQUIRE(index_array == expected_index_array);
  }
}

TEST_CASE("we can construct a table for the multiproduct computation") {
  mtxi::index_table table;

  basct::blob_array term_or_all(0, 8);
  basct::blob_array output_digit_or_all(0, 8);

  SECTION("we handle the empty case") {
    make_multiproduct_table(table, {}, 100, term_or_all, output_digit_or_all, 3);
    REQUIRE(table == mtxi::index_table{});
  }

  SECTION("we handle the case of zero entries") {
    std::vector<uint8_t> exponents = {0};
    std::vector<mtxb::exponent_sequence> sequence = {
        {.element_nbytes = 1, .n = 1, .data = exponents.data()}};
    term_or_all.resize(1, 8);
    output_digit_or_all.resize(1, 8);
    REQUIRE(0 ==
            make_multiproduct_table(table, sequence, 100, term_or_all, output_digit_or_all, 3));
    REQUIRE(table == mtxi::index_table{});
  }

  SECTION("we handle the case of a single exponentiation of 1") {
    std::vector<uint8_t> exponents = {1};
    std::vector<mtxb::exponent_sequence> sequence = {
        {.element_nbytes = 1, .n = 1, .data = exponents.data()}};
    term_or_all.resize(1, 8);
    output_digit_or_all.resize(1, 8);
    term_or_all[0][0] = 1;
    output_digit_or_all[0][0] = 1;
    REQUIRE(1 ==
            make_multiproduct_table(table, sequence, 100, term_or_all, output_digit_or_all, 3));
    REQUIRE(table == mtxi::index_table{{0, 0, 0}});
  }

  SECTION("we handle the case of two exponentiations of 1") {
    std::vector<uint8_t> exponents = {1, 1};
    std::vector<mtxb::exponent_sequence> sequence = {
        {.element_nbytes = 1, .n = exponents.size(), .data = exponents.data()}};
    term_or_all.resize(2, 8);
    term_or_all[0][0] = 1;
    term_or_all[1][0] = 1;
    output_digit_or_all.resize(1, 8);
    output_digit_or_all[0][0] = 1;
    REQUIRE(2 ==
            make_multiproduct_table(table, sequence, 100, term_or_all, output_digit_or_all, 3));
    REQUIRE(table == mtxi::index_table{{0, 0, 0, 1}});
  }

  SECTION("we handle the case of a single exponentiation of 2") {
    std::vector<uint8_t> exponents = {2};
    std::vector<mtxb::exponent_sequence> sequence = {
        {.element_nbytes = 1, .n = 1, .data = exponents.data()}};
    term_or_all.resize(1, 8);
    term_or_all[0][0] = 2;
    output_digit_or_all.resize(1, 8);
    output_digit_or_all[0][0] = 2;
    REQUIRE(1 ==
            make_multiproduct_table(table, sequence, 100, term_or_all, output_digit_or_all, 3));
    REQUIRE(table == mtxi::index_table{{0, 0, 0}});
  }

  SECTION("we handle the case of a single exponentiation of 3") {
    std::vector<uint8_t> exponents = {3};
    std::vector<mtxb::exponent_sequence> sequence = {
        {.element_nbytes = 1, .n = 1, .data = exponents.data()}};
    term_or_all.resize(1, 8);
    term_or_all[0][0] = 3;
    output_digit_or_all.resize(1, 8);
    output_digit_or_all[0][0] = 3;
    REQUIRE(1 ==
            make_multiproduct_table(table, sequence, 100, term_or_all, output_digit_or_all, 3));
    REQUIRE(table == mtxi::index_table{{0, 0, 0}, {1, 0, 0}});
  }

  SECTION("we handle the case of two exponentiations of 1 and 2") {
    std::vector<uint8_t> exponents = {1, 2};
    std::vector<mtxb::exponent_sequence> sequence = {
        {.element_nbytes = 1, .n = exponents.size(), .data = exponents.data()}};
    term_or_all.resize(2, 8);
    term_or_all[0][0] = 1;
    term_or_all[1][0] = 2;
    output_digit_or_all.resize(1, 8);
    output_digit_or_all[0][0] = 0b11;
    REQUIRE(2 ==
            make_multiproduct_table(table, sequence, 100, term_or_all, output_digit_or_all, 3));
    REQUIRE(table == mtxi::index_table{{0, 0, 0}, {1, 0, 1}});
  }

  SECTION("we handle the case of two digits") {
    std::vector<uint8_t> exponents = {3};
    std::vector<mtxb::exponent_sequence> sequence = {
        {.element_nbytes = 1, .n = 1, .data = exponents.data()}};
    term_or_all.resize(1, 8);
    term_or_all[0][0] = 3;
    output_digit_or_all.resize(1, 8);
    output_digit_or_all[0][0] = 1;
    REQUIRE(2 ==
            make_multiproduct_table(table, sequence, 100, term_or_all, output_digit_or_all, 1));
    REQUIRE(table == mtxi::index_table{{0, 0, 0, 1}});
  }

  SECTION("we handle digits sizes of multiple bytes") {
    std::vector<uint16_t> exponents = {0b1010111100110100};
    auto exponents_bytes = reinterpret_cast<uint8_t*>(exponents.data());
    std::vector<mtxb::exponent_sequence> sequence = {{
        .element_nbytes = 2,
        .n = 1,
        .data = exponents_bytes,
    }};

    term_or_all.resize(1, 16);
    term_or_all[0][0] = exponents_bytes[0];
    term_or_all[0][1] = exponents_bytes[1];

    output_digit_or_all.resize(1, 16);
    output_digit_or_all[0][0] = exponents_bytes[0];
    output_digit_or_all[0][1] = exponents_bytes[1];

    REQUIRE(1 ==
            make_multiproduct_table(table, sequence, 100, term_or_all, output_digit_or_all, 16));
    REQUIRE(table == mtxi::index_table{{0, 0, 0},
                                       {1, 0, 0},
                                       {2, 0, 0},
                                       {3, 0, 0},
                                       {4, 0, 0},
                                       {5, 0, 0},
                                       {6, 0, 0},
                                       {7, 0, 0},
                                       {8, 0, 0}});
  }

  SECTION("we handle digits sizes of multiple bytes and multiple digits") {
    std::vector<uint16_t> exponents = {0b1010111100110101};
    auto exponents_bytes = reinterpret_cast<uint8_t*>(exponents.data());
    std::vector<mtxb::exponent_sequence> sequence = {{
        .element_nbytes = 2,
        .n = 1,
        .data = exponents_bytes,
    }};

    term_or_all.resize(1, 16);
    term_or_all[0][0] = exponents_bytes[0];
    term_or_all[0][1] = exponents_bytes[1];

    output_digit_or_all.resize(1, 16);
    output_digit_or_all[0][0] = exponents_bytes[0] | 0b101011;
    output_digit_or_all[0][1] = 0b11;

    REQUIRE(2 ==
            make_multiproduct_table(table, sequence, 100, term_or_all, output_digit_or_all, 10));
    REQUIRE(table == mtxi::index_table{{0, 0, 0, 1},
                                       {1, 0, 1},
                                       {2, 0, 0},
                                       {3, 0, 1},
                                       {4, 0, 0},
                                       {5, 0, 0, 1},
                                       {6, 0, 0},
                                       {7, 0, 0}});
  }
}
