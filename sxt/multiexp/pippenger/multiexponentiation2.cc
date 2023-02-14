#include "sxt/multiexp/pippenger/multiexponentiation2.h"

#include <algorithm>
#include <limits>

#include "sxt/base/bit/span_op.h"
#include "sxt/base/container/blob_array.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/base/digit_utility.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/pippenger/driver2.h"
#include "sxt/multiexp/pippenger/exponent_aggregates.h"
#include "sxt/multiexp/pippenger/exponent_aggregates_computation.h"
#include "sxt/multiexp/pippenger/multiproduct_table.h"

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
static xena::future<memmg::managed_array<void>>
compute_multiproduct(basct::blob_array& output_digit_or_all, const driver2& drv,
                     basct::span_cvoid generators,
                     basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  exponent_aggregates aggregates;
  compute_exponent_aggregates(aggregates, exponents);

  auto radix_log2 =
      aggregates.max_exponent.size() * 8 - basbt::count_leading_zeros(aggregates.max_exponent);
  radix_log2 = std::max(1lu, radix_log2);

  compute_output_digit_or_all(output_digit_or_all, aggregates.output_or_all, radix_log2);

  mtxi::index_table table;
  auto num_multiproduct_inputs =
      make_multiproduct_table(table, exponents, aggregates.pop_count, aggregates.term_or_all,
                              output_digit_or_all, radix_log2);

  return drv.compute_multiproduct(std::move(table), generators, aggregates.term_or_all,
                                  num_multiproduct_inputs);
}

//--------------------------------------------------------------------------------------------------
// compute_multiexponentiation
//--------------------------------------------------------------------------------------------------
xena::future<memmg::managed_array<void>>
compute_multiexponentiation(const driver2& drv, basct::span_cvoid generators,
                            basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  basct::blob_array output_digit_or_all;
  auto multiproduct = compute_multiproduct(output_digit_or_all, drv, generators, exponents);
  return drv.combine_multiproduct_outputs(std::move(multiproduct), std::move(output_digit_or_all));
}
} // namespace sxt::mtxpi
