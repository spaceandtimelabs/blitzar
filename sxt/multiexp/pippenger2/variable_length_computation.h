#pragma once

#include "sxt/base/container/span.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// compute_offset_table 
//--------------------------------------------------------------------------------------------------
void compute_offset_table(basct::span<unsigned> offset_table, basct::cspan<unsigned> bit_table,
                          basct::cspan<unsigned> length_table) noexcept;
} // namespace sxt::mtxpp2
