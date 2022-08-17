#include "sxt/multiexp/pippenger/radix_log2.h"

#include <algorithm>
#include <cmath>

#include "sxt/base/bit/span_op.h"
#include "sxt/base/num/log2p1.h"

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// compute_radix_log2
//--------------------------------------------------------------------------------------------------
size_t compute_radix_log2(basct::cspan<uint8_t> max_exponent, size_t num_inputs,
                          size_t num_outputs) noexcept {

  // check if any of the input values are zero (including the max_exponent)
  if (num_outputs == 0 || num_inputs == 0 ||
      std::all_of(max_exponent.begin(), max_exponent.end(),
                  [](uint8_t b) noexcept { return b == 0; })) {
    return 1;
  }

  if (num_outputs < num_inputs) {
    return max_exponent.size() * 8 - basbt::count_leading_zeros(max_exponent);
  }

  double multiplier_factor;

  if (num_inputs >= num_outputs) {
    multiplier_factor = static_cast<double>(num_outputs) / num_inputs;
  } else {
    multiplier_factor = static_cast<double>(num_inputs) / num_outputs;
  }

  double log_val = sxt::basn::log2p1(max_exponent);

  return std::max(1lu, static_cast<size_t>(std::ceil(std::sqrt(multiplier_factor * log_val))));
}
} // namespace sxt::mtxpi
