#include "sxt/multiexp/pippenger2/variable_length_computation.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can fill in the table of product lengths") {
  std::vector<unsigned> product_lengths_data(10);
  std::vector<unsigned> bit_widths;
  std::vector<unsigned> output_lengths;

  basct::span<unsigned> product_lengths = {product_lengths_data.data(), 1};

  SECTION("we can compute the product length of a single output of a single bit") {
    bit_widths = {1};
    output_lengths = {10};
    compute_product_length_table(product_lengths, bit_widths, output_lengths, 0, 5);
    REQUIRE(product_lengths.size() == 1);
    REQUIRE(product_lengths[0] == 5);
  }

  SECTION("we handle the case when output_length is less than length") {
    bit_widths = {1};
    output_lengths = {10};
    compute_product_length_table(product_lengths, bit_widths, output_lengths, 0, 20);
    REQUIRE(product_lengths.size() == 1);
    REQUIRE(product_lengths[0] == 10);
  }

  SECTION("we handle output of more than a single bit") {
    bit_widths = {2};
    product_lengths = {product_lengths_data.data(), 2};
    output_lengths = {10};
    compute_product_length_table(product_lengths, bit_widths, output_lengths, 0, 5);
    REQUIRE(product_lengths.size() == 2);
    REQUIRE(product_lengths[0] == 5);
    REQUIRE(product_lengths[1] == 5);
  }
}
