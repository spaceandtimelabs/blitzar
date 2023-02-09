#pragma once

#include <array>
#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::basct {
class blob_array;
}
namespace sxt::mtxb {
class exponent_sequence;
} // namespace sxt::mtxb
namespace sxt::mtxi {
class index_table;
}

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// make_digit_index_array
//--------------------------------------------------------------------------------------------------
void make_digit_index_array(basct::span<size_t> array, size_t first,
                            basct::cspan<uint8_t> or_all) noexcept;

//--------------------------------------------------------------------------------------------------
// make_multiproduct_term_table
//--------------------------------------------------------------------------------------------------
size_t make_multiproduct_term_table(mtxi::index_table& table, const basct::blob_array& term_or_all,
                                    size_t radix_log2) noexcept;

//--------------------------------------------------------------------------------------------------
// make_multiproduct_table
//--------------------------------------------------------------------------------------------------
size_t make_multiproduct_table(mtxi::index_table& table,
                               basct::cspan<mtxb::exponent_sequence> exponents, size_t max_entries,
                               const basct::blob_array& term_or_all,
                               const basct::blob_array& output_digit_or_all,
                               size_t radix_log2) noexcept;
} // namespace sxt::mtxpi
