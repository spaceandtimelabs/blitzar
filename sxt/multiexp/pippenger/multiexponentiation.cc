#include "sxt/multiexp/pippenger/multiexponentiation.h"

#include <vector>

#include "sxt/base/bit/count.h"
#include "sxt/base/bit/span_op.h"
#include "sxt/base/container/blob_array.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/multiexp/base/digit_utility.h"
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
static void compute_output_digit_or_all(basct::blob_array& output_digit_or_all,
                                        const basct::blob_array& output_or_all,
                                        size_t radix_log2) noexcept {
  auto digit_num_bytes = basn::divide_up(radix_log2, 8ul);
  auto num_outputs = output_or_all.size();
  output_digit_or_all.resize(num_outputs, digit_num_bytes);
  std::vector<uint8_t> digit(digit_num_bytes);
  for (size_t output_index = 0; output_index < num_outputs; ++output_index) {
    auto or_all = output_or_all[output_index];
    auto num_digits = mtxb::count_num_digits(or_all, radix_log2);
    auto digit_or_all = output_digit_or_all[output_index];
    for (size_t digit_index = 0; digit_index < num_digits; ++digit_index) {
      mtxb::extract_digit(digit, or_all, radix_log2, digit_index);
      basbt::or_equal(digit_or_all, digit);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// compute_multiproduct
//--------------------------------------------------------------------------------------------------
static void compute_multiproduct(memmg::managed_array<void>& inout,
                                 basct::blob_array& output_digit_or_all, const driver& drv,
                                 basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  exponent_aggregates aggregates;
  compute_exponent_aggregates(aggregates, exponents);

  size_t radix_log2 = mtxpi::compute_radix_log2(
      aggregates.max_exponent, aggregates.term_or_all.size(), aggregates.output_or_all.size());

  mtxi::index_table table;
  auto num_multiproduct_inputs =
      make_multiproduct_term_table(table, aggregates.term_or_all, radix_log2);

  drv.compute_multiproduct_inputs(inout, table.cheader(), radix_log2, num_multiproduct_inputs,
                                  aggregates.pop_count);

  compute_output_digit_or_all(output_digit_or_all, aggregates.output_or_all, radix_log2);

  make_multiproduct_table(table, exponents, aggregates.pop_count, aggregates.term_or_all,
                          output_digit_or_all, radix_log2);

  drv.compute_multiproduct(inout, table, num_multiproduct_inputs);
}

//--------------------------------------------------------------------------------------------------
// compute_multiexponentiation
//--------------------------------------------------------------------------------------------------
void compute_multiexponentiation(memmg::managed_array<void>& inout, const driver& drv,
                                 basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  basct::blob_array output_digit_or_all;
  compute_multiproduct(inout, output_digit_or_all, drv, exponents);

  basct::span<uint8_t> output_digit_or_all_p{output_digit_or_all.data(),
                                             output_digit_or_all.size()};
  drv.combine_multiproduct_outputs(inout, output_digit_or_all);
}
} // namespace sxt::mtxpi
