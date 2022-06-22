#include "sxt/multiexp/pippenger/multiproduct_table.h"

#include "sxt/base/bit/count.h"
#include "sxt/base/bit/iteration.h"
#include "sxt/multiexp/base/digit_utility.h"
#include "sxt/multiexp/base/exponent.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/base/exponent_utility.h"
#include "sxt/multiexp/index/index_table.h"

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// init_multiproduct_table
//--------------------------------------------------------------------------------------------------
static void init_multiproduct_table(
    mtxi::index_table& table, size_t max_entries,
    basct::cspan<uint8_t> output_digit_or_all) noexcept {
  size_t row_count = 0;
  for (auto& digit : output_digit_or_all) {
    row_count += basbt::pop_count(digit);
  }
  table.reshape(row_count, max_entries);
}

//--------------------------------------------------------------------------------------------------
// init_multiproduct_output_rows
//--------------------------------------------------------------------------------------------------
static uint64_t* init_multiproduct_output_rows(
    basct::span<basct::span<uint64_t>> rows, size_t& multiproduct_output_index,
    uint64_t* entry_data, const mtxb::exponent_sequence& sequence,
    uint8_t digit_or_all, size_t radix_log2) noexcept {
  std::array<size_t, 8> row_counts = {};
  for (size_t term_index = 0; term_index < sequence.n; ++term_index) {
    basct::cspan<uint8_t> e{
        sequence.data + term_index * sequence.element_nbytes,
        sequence.element_nbytes};
    auto num_digits = mtxb::count_num_digits(e, radix_log2);
    for(size_t digit_index=0; digit_index<num_digits; ++digit_index) {
      uint64_t digit = mtxb::extract_digit(e, radix_log2, digit_index);
      while (digit != 0) {
        auto pos = basbt::consume_next_bit(digit);
        ++row_counts[pos];
      }
    }
  }
  for (auto count : row_counts) {
    if (count == 0) {
      continue;
    }
    rows[multiproduct_output_index++] = {entry_data, 0};
    entry_data += count;
  }
  return entry_data;
}

//--------------------------------------------------------------------------------------------------
// make_digit_index_array
//--------------------------------------------------------------------------------------------------
void make_digit_index_array(std::array<size_t, 8>& array, size_t first,
                            uint8_t or_all) noexcept {
  array = {};
  uint64_t x = or_all;
  while (x != 0) {
    auto pos = basbt::consume_next_bit(x);
    array[pos] = first++;
  }
}

//--------------------------------------------------------------------------------------------------
// make_multiproduct_term_table
//--------------------------------------------------------------------------------------------------
void make_multiproduct_term_table(mtxi::index_table& table,
                                  basct::cspan<mtxb::exponent> term_or_all,
                                  size_t radix_log2) noexcept {
  size_t num_nonzero_digits = 0;
  for (auto& e : term_or_all) {
    num_nonzero_digits += mtxb::count_nonzero_digits(e, radix_log2);
  }
  table.reshape(term_or_all.size(), num_nonzero_digits);
  auto rows = table.header();
  auto entries = table.entry_data();
  for (size_t term_index = 0; term_index < term_or_all.size(); ++term_index) {
    auto& e = term_or_all[term_index];
    size_t entry_count = 0;
    auto num_digits = mtxb::count_num_digits(e, radix_log2);
    for (size_t digit_index = 0; digit_index < num_digits; ++digit_index) {
      auto digit = mtxb::extract_digit(e, radix_log2, digit_index);
      if (digit != 0) {
        entries[entry_count++] = digit_index;
      }
    }
    rows[term_index] = {entries, entry_count};
    entries += entry_count;
  }
}

//--------------------------------------------------------------------------------------------------
// make_multiproduct_table
//--------------------------------------------------------------------------------------------------
void make_multiproduct_table(mtxi::index_table& table,
                             basct::cspan<mtxb::exponent_sequence> exponents,
                             size_t max_entries,
                             basct::cspan<mtxb::exponent> term_or_all,
                             basct::cspan<uint8_t> output_digit_or_all,
                             size_t radix_log2) noexcept {
  assert(exponents.size() == output_digit_or_all.size());

  init_multiproduct_table(table, max_entries, output_digit_or_all);

  auto entry_data = table.entry_data();
  auto rows = table.header();
  size_t multiproduct_output_index = 0;
  for (size_t output_index = 0; output_index < output_digit_or_all.size();
       ++output_index) {
    auto digit_or_all = output_digit_or_all[output_index];
    auto sequence = exponents[output_index];
    std::array<size_t, 8> index_array;
    make_digit_index_array(index_array, multiproduct_output_index,
                           digit_or_all);
    entry_data = init_multiproduct_output_rows(rows, multiproduct_output_index,
                                               entry_data, sequence,
                                               digit_or_all, radix_log2);
    size_t input_first = 0;
    for (size_t term_index = 0; term_index < sequence.n; ++term_index) {
      basct::cspan<uint8_t> e{
          sequence.data + term_index * sequence.element_nbytes,
          sequence.element_nbytes};
      auto num_digits = mtxb::count_num_digits(e, radix_log2);
      size_t input_offset = 0;
      for (size_t digit_index = 0; digit_index < num_digits; ++digit_index) {
        uint64_t digit = mtxb::extract_digit(e, radix_log2, digit_index);
        while (digit != 0) {
          auto pos = basbt::consume_next_bit(digit);
          auto& row = rows[index_array[pos]];
          row[row.size()] = input_first + input_offset;
          row = {row.data(), row.size() + 1};
        }
        input_offset += static_cast<size_t>(
            mtxb::extract_digit(term_or_all[term_index], radix_log2,
                                digit_index) != 0);
      }
      input_first +=
          mtxb::count_nonzero_digits(term_or_all[term_index], radix_log2);
    }
  }
}
}  // namespace sxt::mtxpi
