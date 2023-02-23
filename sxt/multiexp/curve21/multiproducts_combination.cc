#include "sxt/multiexp/curve21/multiproducts_combination.h"

#include "sxt/base/bit/span_op.h"
#include "sxt/base/container/blob_array.h"
#include "sxt/base/error/assert.h"
#include "sxt/curve21/constant/zero.h"
#include "sxt/multiexp/curve21/doubling_reduction.h"

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// combine_multiproducts
//--------------------------------------------------------------------------------------------------
void combine_multiproducts(basct::span<c21t::element_p3> outputs,
                           const basct::blob_array& output_digit_or_all,
                           basct::cspan<c21t::element_p3> products) noexcept {
  size_t input_index = 0;
  for (size_t output_index = 0; output_index < output_digit_or_all.size(); ++output_index) {
    auto digit_or_all = output_digit_or_all[output_index];
    int digit_count_one = basbt::pop_count(digit_or_all);
    if (digit_count_one == 0) {
      outputs[output_index] = c21cn::zero_p3_v;
      continue;
    }
    c21t::element_p3 output;
    SXT_DEBUG_ASSERT(input_index + digit_count_one <= products.size());
    doubling_reduce(output, digit_or_all,
                    basct::cspan<c21t::element_p3>{&products[input_index],
                                                   static_cast<size_t>(digit_count_one)});
    input_index += digit_count_one;
    outputs[output_index] = output;
  }
}
} // namespace sxt::mtxc21
