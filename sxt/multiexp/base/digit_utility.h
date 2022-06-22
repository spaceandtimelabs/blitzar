#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// extract_digit
//--------------------------------------------------------------------------------------------------
uint8_t extract_digit(basct::cspan<uint8_t> e, size_t radix_log2,
                      size_t digit_index) noexcept;

//--------------------------------------------------------------------------------------------------
// count_nonzero_digits
//--------------------------------------------------------------------------------------------------
size_t count_nonzero_digits(basct::cspan<uint8_t> e, size_t highest_bit,
                            size_t radix_log2) noexcept;

//--------------------------------------------------------------------------------------------------
// count_num_digits
//--------------------------------------------------------------------------------------------------
size_t count_num_digits(basct::cspan<uint8_t> e, size_t radix_log2) noexcept;
} // namespace sxt::mtxb
