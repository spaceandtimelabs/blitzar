/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "sxt/multiexp/base/digit_utility.h"

#include "sxt/base/bit/count.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/divide_up.h"

namespace sxt::mtxb {
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
