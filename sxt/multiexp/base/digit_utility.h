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
#pragma once

#include <cstdint>
#include <cassert>

#include "sxt/base/container/span.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/num/divide_up.h"

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// extract_digit
//--------------------------------------------------------------------------------------------------
inline CUDA_CALLABLE void extract_digit(basct::span<uint8_t> digit, basct::cspan<uint8_t> e,
                                        size_t radix_log2, size_t digit_index) noexcept {
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
bool is_digit_zero(basct::cspan<uint8_t> e, size_t radix_log2, size_t digit_index) noexcept;

//--------------------------------------------------------------------------------------------------
// get_last_digit
//--------------------------------------------------------------------------------------------------
size_t get_last_digit(basct::cspan<uint8_t> e, size_t radix_log2) noexcept;

//--------------------------------------------------------------------------------------------------
// count_nonzero_digits
//--------------------------------------------------------------------------------------------------
size_t count_nonzero_digits(basct::cspan<uint8_t> e, size_t radix_log2) noexcept;

//--------------------------------------------------------------------------------------------------
// count_num_digits
//--------------------------------------------------------------------------------------------------
size_t count_num_digits(basct::cspan<uint8_t> e, size_t radix_log2) noexcept;
} // namespace sxt::mtxb
