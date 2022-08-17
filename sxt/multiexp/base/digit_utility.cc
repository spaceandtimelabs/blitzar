#include "sxt/multiexp/base/digit_utility.h"

#include <cassert>

#include "sxt/base/bit/count.h"
#include "sxt/base/num/divide_up.h"

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// extract_digit
//--------------------------------------------------------------------------------------------------
void extract_digit(basct::span<uint8_t> digit, basct::cspan<uint8_t> e, size_t radix_log2,
                   size_t digit_index) noexcept {
  assert(digit.size() == basn::divide_up(radix_log2, 8ul));
  auto digit_num_bytes = digit.size();
  auto bit_first = radix_log2 * digit_index;
  assert(e.size() * 8 > bit_first);
  auto byte_first = bit_first / 8;
  auto offset = bit_first - 8 * byte_first;
  digit[0] = e[byte_first] >> offset;
  auto byte_last = std::min(digit_num_bytes, e.size() - byte_first);
  for (size_t byte_index = 1; byte_index < byte_last; ++byte_index) {
    auto byte = e[byte_first + byte_index];
    digit[byte_index - 1] |= byte << (8 - offset);
    digit[byte_index] = byte >> offset;
  }
  for (size_t byte_index = byte_last; byte_index < digit_num_bytes; ++byte_index) {
    digit[byte_index] = 0;
  }
  if (offset + radix_log2 > 8 && byte_first + digit_num_bytes < e.size()) {
    digit[digit_num_bytes - 1] |= e[byte_first + digit_num_bytes] << (8 - offset);
  }
  constexpr uint8_t ones = 0xff;
  digit[digit_num_bytes - 1] &= ones >> (digit_num_bytes * 8 - radix_log2);
}

//--------------------------------------------------------------------------------------------------
// is_digit_zero
//--------------------------------------------------------------------------------------------------
bool is_digit_zero(basct::cspan<uint8_t> e, size_t radix_log2, size_t digit_index) noexcept {
  auto bit_first = radix_log2 * digit_index;
  auto i = bit_first / 8;
  auto offset = bit_first - 8 * i;
  uint8_t b = e[i] & (static_cast<uint8_t>(-1) << offset);
  size_t count = radix_log2 + offset;
  while (count >= 8) {
    if (b != 0) {
      return false;
    }
    ++i;
    if (i == e.size()) {
      return true;
    }
    b = e[i];
    count -= 8;
  }
  return static_cast<uint8_t>(b << (8 - count)) == 0;
}

//--------------------------------------------------------------------------------------------------
// get_last_digit
//--------------------------------------------------------------------------------------------------
size_t get_last_digit(basct::cspan<uint8_t> e, size_t radix_log2) noexcept {
  auto leading_zeros = basbt::count_leading_zeros(e.data(), e.size());
  return basn::divide_up(e.size() * 8 - leading_zeros, radix_log2);
}

//--------------------------------------------------------------------------------------------------
// count_nonzero_digits
//--------------------------------------------------------------------------------------------------
size_t count_nonzero_digits(basct::cspan<uint8_t> e, size_t radix_log2) noexcept {
  size_t res = 0;
  auto last_digit = get_last_digit(e, radix_log2);
  for (size_t digit_index = 0; digit_index < last_digit; ++digit_index) {
    res += static_cast<size_t>(!is_digit_zero(e, radix_log2, digit_index));
  }
  return res;
}

//--------------------------------------------------------------------------------------------------
// count_num_digits
//--------------------------------------------------------------------------------------------------
size_t count_num_digits(basct::cspan<uint8_t> e, size_t radix_log2) noexcept {
  auto t = e.size() * 8 - basbt::count_leading_zeros(e.data(), e.size());
  return basn::divide_up(t, radix_log2);
}
} // namespace sxt::mtxb
