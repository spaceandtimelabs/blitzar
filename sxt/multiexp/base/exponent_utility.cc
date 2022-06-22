#include "sxt/multiexp/base/exponent_utility.h"

#include <cassert>

#include "sxt/multiexp/base/digit_utility.h"
#include "sxt/multiexp/base/exponent.h"

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// or_equal
//--------------------------------------------------------------------------------------------------
void or_equal(exponent& e, basct::cspan<uint8_t> value) noexcept {
  assert(value.size() <= 32);
  auto e_data = e.data();
  for (size_t i=0; i<value.size(); ++i) {
    e_data[i] |= value[i];
  }
}

//--------------------------------------------------------------------------------------------------
// count_nonzero_digits
//--------------------------------------------------------------------------------------------------
size_t count_nonzero_digits(const exponent& e, size_t radix_log2) noexcept {
  auto highest_bit = e.highest_bit();
  if (highest_bit < 0) {
    return 0;
  }
  return count_nonzero_digits({e.data(), 32}, static_cast<size_t>(highest_bit),
                              radix_log2);
}

//--------------------------------------------------------------------------------------------------
// count_num_digits
//--------------------------------------------------------------------------------------------------
size_t count_num_digits(const exponent& e, size_t radix_log2) noexcept {
  return count_num_digits({e.data(), 32}, radix_log2);
}

//--------------------------------------------------------------------------------------------------
// extract_digit
//--------------------------------------------------------------------------------------------------
uint8_t extract_digit(const exponent& e, size_t radix_log2,
                      size_t digit_index) noexcept {
  return extract_digit({e.data(), 32}, radix_log2, digit_index);
}
}  // namespace sxt::mtxb
