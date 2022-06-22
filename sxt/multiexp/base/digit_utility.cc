#include "sxt/multiexp/base/digit_utility.h"

#include <cassert>

#include "sxt/base/bit/count.h"
#include "sxt/base/num/divide_up.h"

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// extract_digit
//--------------------------------------------------------------------------------------------------
uint8_t extract_digit(basct::cspan<uint8_t> e, size_t radix_log2,
                      size_t digit_index) noexcept {
  assert(radix_log2 <= 8);
  auto bit_first = radix_log2 * digit_index;
  assert(e.size() * 8 > bit_first);
  auto byte_first = bit_first / 8;
  auto offset = bit_first - 8 * byte_first;
  auto res = e[byte_first];
  res >>= offset;
  if (offset + radix_log2 > 8 && byte_first < e.size() - 1) {
    res |= e[byte_first + 1] << (8 - offset);
  }
  constexpr uint8_t ones = 0xff;
  return res & (ones >> (8 - radix_log2));
}

//--------------------------------------------------------------------------------------------------
// count_nonzero_digits
//--------------------------------------------------------------------------------------------------
size_t count_nonzero_digits(basct::cspan<uint8_t> e, size_t highest_bit,
                            size_t radix_log2) noexcept {
  assert(highest_bit < e.size() * 8);
  size_t res = 0;
  for (size_t digit_index = 0; radix_log2 * digit_index <= highest_bit;
       ++digit_index) {
    auto digit = extract_digit(e, radix_log2, digit_index);
    res += static_cast<size_t>(digit != 0);
  }
  return res;
}

//--------------------------------------------------------------------------------------------------
// count_num_digits
//--------------------------------------------------------------------------------------------------
size_t count_num_digits(basct::cspan<uint8_t> e, size_t radix_log2) noexcept {
  auto highest_bit =
      e.size() * 8 - basbt::count_leading_zeros(e.data(), e.size());
  return basn::divide_up(highest_bit + 1, radix_log2);
}
} // namespace sxt::mtxb
