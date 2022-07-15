#include "sxt/multiexp/pippenger/multiexponentiation.h"

#include <vector>

#include "sxt/base/bit/count.h"

#include "sxt/multiexp/base/exponent_utility.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/pippenger/driver.h"
#include "sxt/multiexp/pippenger/exponent_aggregates.h"
#include "sxt/multiexp/pippenger/exponent_aggregates_computation.h"
#include "sxt/multiexp/pippenger/multiproduct_table.h"
#include "sxt/multiexp/pippenger/radix_log2.h"

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// compute_output_digit_or_all
//--------------------------------------------------------------------------------------------------
static void compute_output_digit_or_all(std::vector<uint8_t>& output_digit_or_all,
                                        basct::cspan<mtxb::exponent> output_or_all,
                                        size_t radix_log2) noexcept {
  output_digit_or_all.reserve(output_or_all.size());
  for (size_t output_index = 0; output_index < output_or_all.size(); ++output_index) {
    auto& or_all = output_or_all[output_index];
    auto num_digits = mtxb::count_num_digits(or_all, radix_log2);
    uint8_t mask = 0;
    for (size_t digit_index = 0; digit_index < num_digits; ++digit_index) {
      auto digit = mtxb::extract_digit(or_all, radix_log2, digit_index);
      mask |= digit;
    }
    output_digit_or_all.push_back(mask);
  }
}

//--------------------------------------------------------------------------------------------------
// compute_multiproduct
//--------------------------------------------------------------------------------------------------
static void compute_multiproduct(memmg::managed_array<void>& inout,
                                 std::vector<uint8_t>& output_digit_or_all, const driver& drv,
                                 basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  exponent_aggregates aggregates;
  compute_exponent_aggregates(aggregates, exponents);

  size_t radix_log2 = std::min(8ul, mtxpi::compute_radix_log2(aggregates.max_exponent,
                                                              aggregates.term_or_all.size(),
                                                              aggregates.output_or_all.size()));

  mtxi::index_table table;
  make_multiproduct_term_table(table, aggregates.term_or_all, radix_log2);

  drv.compute_multiproduct_inputs(inout, table.cheader(), radix_log2);

  compute_output_digit_or_all(output_digit_or_all, aggregates.output_or_all, radix_log2);

  make_multiproduct_table(table, exponents, aggregates.pop_count, aggregates.term_or_all,
                          output_digit_or_all, radix_log2);

  drv.compute_multiproduct(inout, table);
}

//--------------------------------------------------------------------------------------------------
// compute_multiexponentiation
//--------------------------------------------------------------------------------------------------
void compute_multiexponentiation(memmg::managed_array<void>& inout, const driver& drv,
                                 basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  std::vector<uint8_t> output_digit_or_all;
  compute_multiproduct(inout, output_digit_or_all, drv, exponents);
  drv.combine_multiproduct_outputs(inout, output_digit_or_all);
}
} // namespace sxt::mtxpi
