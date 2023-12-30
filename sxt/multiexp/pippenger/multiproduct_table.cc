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
#include "sxt/multiexp/pippenger/multiproduct_table.h"

#include <algorithm>

#include "sxt/base/bit/count.h"
#include "sxt/base/bit/iteration.h"
#include "sxt/base/bit/span_op.h"
#include "sxt/base/container/blob_array.h"
#include "sxt/base/container/span_utility.h"
#include "sxt/base/container/stack_array.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/abs.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/base/num/constexpr_switch.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/base/num/power2_equality.h"
#include "sxt/base/type/int.h"
#include "sxt/multiexp/base/digit_utility.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/index/index_table.h"

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// init_multiproduct_table
//--------------------------------------------------------------------------------------------------
static void init_multiproduct_table(mtxi::index_table& table, size_t max_entries,
                                    const basct::blob_array& output_digit_or_all) noexcept {
  size_t row_count = 0;
  for (auto digit : output_digit_or_all) {
    row_count += basbt::pop_count(digit);
  }
  table.reshape(row_count, max_entries + 2 * row_count);
}

//--------------------------------------------------------------------------------------------------
// init_multiproduct_output_rows_impl
//--------------------------------------------------------------------------------------------------
static uint64_t* init_multiproduct_output_rows_impl(basct::span<basct::span<uint64_t>> rows,
                                                    size_t& multiproduct_output_index,
                                                    uint64_t* entry_data,
                                                    basct::span<size_t> row_counts) noexcept {
  for (auto count : row_counts) {
    if (count == 0) {
      continue;
    }
    auto output_index = multiproduct_output_index++;
    auto& row = rows[output_index];
    row = {entry_data, 2};
    row[0] = output_index;
    row[1] = 0;
    entry_data += count + 2;
  }
  return entry_data;
}

//--------------------------------------------------------------------------------------------------
// init_multiproduct_output_rows
//--------------------------------------------------------------------------------------------------
static uint64_t* init_multiproduct_output_rows(basct::span<basct::span<uint64_t>> rows,
                                               size_t& multiproduct_output_index,
                                               uint64_t* entry_data, basct::span<size_t> row_counts,
                                               const mtxb::exponent_sequence& sequence,
                                               size_t digit_num_bytes) noexcept {
  SXT_STACK_ARRAY(digit, digit_num_bytes, uint8_t);
  std::fill(row_counts.begin(), row_counts.end(), 0);
  auto radix_log2 = row_counts.size();
  for (size_t term_index = 0; term_index < sequence.n; ++term_index) {
    basct::cspan<uint8_t> e{sequence.data + term_index * sequence.element_nbytes,
                            sequence.element_nbytes};
    auto digit_last = mtxb::get_last_digit(e, radix_log2);
    for (size_t digit_index = 0; digit_index < digit_last; ++digit_index) {
      mtxb::extract_digit(digit, e, radix_log2, digit_index);
      basbt::for_each_bit(digit, [&](size_t bit_index) noexcept { ++row_counts[bit_index]; });
    }
  }
  return init_multiproduct_output_rows_impl(rows, multiproduct_output_index, entry_data,
                                            row_counts);
}

//--------------------------------------------------------------------------------------------------
// init_signed_multiproduct_output_rows
//--------------------------------------------------------------------------------------------------
template <size_t NumBytes>
static uint64_t* init_signed_multiproduct_output_rows(basct::span<basct::span<uint64_t>> rows,
                                                      size_t& multiproduct_output_index,
                                                      uint64_t* entry_data,
                                                      basct::span<size_t> row_counts,
                                                      const mtxb::exponent_sequence& sequence,
                                                      size_t digit_num_bytes) noexcept {
  SXT_STACK_ARRAY(digit, digit_num_bytes, uint8_t);
  std::fill(row_counts.begin(), row_counts.end(), 0);
  auto radix_log2 = row_counts.size() / 2u;
  for (size_t term_index = 0; term_index < sequence.n; ++term_index) {
    basct::cspan<uint8_t> e{sequence.data + term_index * sequence.element_nbytes,
                            sequence.element_nbytes};
    bast::sized_int_t<NumBytes * 8> x;
    std::copy(e.begin(), e.end(), reinterpret_cast<uint8_t*>(&x));
    auto abs_x = basn::abs(x);
    e = {reinterpret_cast<uint8_t*>(&abs_x), NumBytes};
    auto offset = static_cast<size_t>(abs_x != x) * radix_log2;
    auto digit_last = mtxb::get_last_digit(e, radix_log2);
    for (size_t digit_index = 0; digit_index < digit_last; ++digit_index) {
      mtxb::extract_digit(digit, e, radix_log2, digit_index);
      basbt::for_each_bit(digit,
                          [&](size_t bit_index) noexcept { ++row_counts[offset + bit_index]; });
    }
  }
  return init_multiproduct_output_rows_impl(rows, multiproduct_output_index, entry_data,
                                            row_counts);
}

//--------------------------------------------------------------------------------------------------
// fill_from_sequence
//--------------------------------------------------------------------------------------------------
static size_t fill_from_sequence(uint64_t*& entry_data, basct::span<basct::span<uint64_t>> rows,
                                 size_t& multiproduct_output_index,
                                 const mtxb::exponent_sequence& sequence,
                                 basct::cspan<uint8_t> digit_or_all,
                                 const basct::blob_array& term_or_all, size_t radix_log2) noexcept {
  auto digit_num_bytes = basn::divide_up(radix_log2, 8ul);
  SXT_STACK_ARRAY(index_array, radix_log2, size_t);
  auto bit_index_first = multiproduct_output_index;
  entry_data = init_multiproduct_output_rows(rows, multiproduct_output_index, entry_data,
                                             index_array, sequence, digit_num_bytes);
  make_digit_index_array(index_array, bit_index_first, digit_or_all);
  size_t input_first = 0;
  SXT_STACK_ARRAY(digit, digit_num_bytes, uint8_t);
  for (size_t term_index = 0; term_index < sequence.n; ++term_index) {
    basct::cspan<uint8_t> e{sequence.data + term_index * sequence.element_nbytes,
                            sequence.element_nbytes};
    auto digit_last = mtxb::get_last_digit(e, radix_log2);
    size_t input_offset = 0;
    for (size_t digit_index = 0; digit_index < digit_last; ++digit_index) {
      mtxb::extract_digit(digit, e, radix_log2, digit_index);
      basbt::for_each_bit(digit, [&](size_t bit_index) noexcept {
        auto& row = rows[index_array[bit_index]];
        auto sz = row.size();
        row = {row.data(), row.size() + 1};
        row[sz] = input_first + input_offset;
      });
      input_offset += static_cast<size_t>(
          !mtxb::is_digit_zero(term_or_all[term_index], radix_log2, digit_index));
    }
    input_first += mtxb::count_nonzero_digits(term_or_all[term_index], radix_log2);
  }
  return input_first;
}

//--------------------------------------------------------------------------------------------------
// fill_from_signed_sequence
//--------------------------------------------------------------------------------------------------
template <size_t NumBytes>
static size_t fill_from_signed_sequence(
    uint64_t*& entry_data, basct::span<basct::span<uint64_t>> rows,
    size_t& multiproduct_output_index, const mtxb::exponent_sequence& sequence,
    basct::cspan<uint8_t> pos_digit_or_all, basct::cspan<uint8_t> neg_digit_or_all,
    const basct::blob_array& term_or_all, size_t radix_log2) noexcept {
  auto digit_num_bytes = basn::divide_up(radix_log2, 8ul);
  SXT_STACK_ARRAY(index_array, radix_log2 * 2u, size_t);
  auto bit_index_first = multiproduct_output_index;
  entry_data = init_signed_multiproduct_output_rows<NumBytes>(
      rows, multiproduct_output_index, entry_data, index_array, sequence, digit_num_bytes);
  auto pos_bit_index_last = make_digit_index_array(basct::subspan(index_array, 0, radix_log2),
                                                   bit_index_first, pos_digit_or_all);
  make_digit_index_array(basct::subspan(index_array, radix_log2), pos_bit_index_last,
                         neg_digit_or_all);
  size_t input_first = 0;
  SXT_STACK_ARRAY(digit, digit_num_bytes, uint8_t);
  for (size_t term_index = 0; term_index < sequence.n; ++term_index) {
    basct::cspan<uint8_t> e{sequence.data + term_index * sequence.element_nbytes,
                            sequence.element_nbytes};
    bast::sized_int_t<NumBytes * 8> x;
    std::copy(e.begin(), e.end(), reinterpret_cast<uint8_t*>(&x));
    auto abs_x = basn::abs(x);
    e = {reinterpret_cast<uint8_t*>(&abs_x), NumBytes};
    auto digit_last = mtxb::get_last_digit(e, radix_log2);
    size_t input_offset = 0;
    auto bit_index_offset = static_cast<size_t>(x != abs_x) * radix_log2;
    for (size_t digit_index = 0; digit_index < digit_last; ++digit_index) {
      mtxb::extract_digit(digit, e, radix_log2, digit_index);
      basbt::for_each_bit(digit, [&](size_t bit_index) noexcept {
        auto& row = rows[index_array[bit_index + bit_index_offset]];
        auto sz = row.size();
        row = {row.data(), row.size() + 1};
        row[sz] = input_first + input_offset;
      });
      input_offset += static_cast<size_t>(
          !mtxb::is_digit_zero(term_or_all[term_index], radix_log2, digit_index));
    }
    input_first += mtxb::count_nonzero_digits(term_or_all[term_index], radix_log2);
  }
  return input_first;
}

//--------------------------------------------------------------------------------------------------
// make_digit_index_array
//--------------------------------------------------------------------------------------------------
size_t make_digit_index_array(basct::span<size_t> array, size_t first,
                              basct::cspan<uint8_t> or_all) noexcept {
  basbt::for_each_bit(or_all, [&](size_t index) noexcept { array[index] = first++; });
  return first;
}

//--------------------------------------------------------------------------------------------------
// make_multiproduct_table
//--------------------------------------------------------------------------------------------------
size_t make_multiproduct_table(mtxi::index_table& table,
                               basct::cspan<mtxb::exponent_sequence> exponents, size_t max_entries,
                               const basct::blob_array& term_or_all,
                               const basct::blob_array& output_digit_or_all,
                               size_t radix_log2) noexcept {
  init_multiproduct_table(table, max_entries, output_digit_or_all);

  auto entry_data = table.entry_data();
  auto rows = table.header();
  size_t multiproduct_output_index = 0;
  size_t max_inputs = 0;
  size_t input_first;
  size_t output_index = 0;
  for (size_t sequence_index = 0; sequence_index < exponents.size(); ++sequence_index) {
    auto sequence = exponents[sequence_index];
    auto element_num_bytes = sequence.element_nbytes;
    if (sequence.is_signed) {
      SXT_DEBUG_ASSERT(basn::is_power2(element_num_bytes));
      SXT_RELEASE_ASSERT(element_num_bytes <= 16,
                         "signed commitments for numbers larger than 128-bits aren't supported");
      basn::constexpr_switch<5>(
          basn::ceil_log2(element_num_bytes),
          [&]<unsigned NumBytesLg2>(std::integral_constant<unsigned, NumBytesLg2>) noexcept {
            static constexpr auto NumBytes = 1ull << NumBytesLg2;
            input_first = fill_from_signed_sequence<NumBytes>(
                entry_data, rows, multiproduct_output_index, sequence,
                output_digit_or_all[output_index], output_digit_or_all[output_index + 1],
                term_or_all, radix_log2);
          });
      output_index += 2;
    } else {
      input_first = fill_from_sequence(entry_data, rows, multiproduct_output_index, sequence,
                                       output_digit_or_all[output_index], term_or_all, radix_log2);
      ++output_index;
    }
    max_inputs = std::max(input_first, max_inputs);
  }
  return max_inputs;
}
} // namespace sxt::mtxpi
