#include "sxt/multiexp/pippenger2/variable_length_computation.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can fill in the table of product lengths") {
  std::vector<unsigned> product_lengths_data(1);
  std::vector<unsigned> bit_widths;
  std::vector<unsigned> output_lengths;

  basct::span<unsigned> product_lengths = product_lengths_data;

  SECTION("we can compute the product length of a single output of a single bit") {
    bit_widths = {1};
    output_lengths = {10};
    compute_product_length_table(product_lengths, bit_widths, output_lengths, 0, 5);
    REQUIRE(product_lengths.size() == 1);
    REQUIRE(product_lengths[0] == 5);
  }
/* void compute_product_length_table(basct::span<unsigned>& product_lengths, basct::cspan<unsigned> bit_widths, */
/*                                   basct::cspan<unsigned> output_lengths, unsigned first, */
/*                                   unsigned length) noexcept; */
}
