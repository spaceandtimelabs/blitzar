#pragma once

#include "sxt/base/container/span.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// compute_product_length_table 
//--------------------------------------------------------------------------------------------------
void compute_product_length_table(basct::span<unsigned>& product_lengths, basct::cspan<unsigned> bit_widths,
                                  basct::cspan<unsigned> output_lengths, unsigned first,
                                  unsigned length) noexcept;
} // namespace sxt::mtxpp2
