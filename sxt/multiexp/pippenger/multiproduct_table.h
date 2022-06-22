#pragma once

#include <array>
#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::mtxb { class exponent; }
namespace sxt::mtxb { class exponent_sequence; }
namespace sxt::mtxi { class index_table; }

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// make_digit_index_array
//--------------------------------------------------------------------------------------------------
void make_digit_index_array(std::array<size_t, 8>& array, size_t first,
                            uint8_t or_all) noexcept;

//--------------------------------------------------------------------------------------------------
// make_multiproduct_term_table
//--------------------------------------------------------------------------------------------------
void make_multiproduct_term_table(mtxi::index_table& table,
                                  basct::cspan<mtxb::exponent> term_or_all,
                                  size_t radix_log2) noexcept;

//--------------------------------------------------------------------------------------------------
// make_multiproduct_table
//--------------------------------------------------------------------------------------------------
void make_multiproduct_table(mtxi::index_table& table,
                             basct::cspan<mtxb::exponent_sequence> exponents,
                             size_t max_entries,
                             basct::cspan<mtxb::exponent> term_or_all,
                             basct::cspan<uint8_t> output_digit_or_all,
                             size_t radix_log2) noexcept;
}  // namespace sxt::mtxpi
