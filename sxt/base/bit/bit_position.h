#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::basbt {
//--------------------------------------------------------------------------------------------------
// compute_bit_positions
//--------------------------------------------------------------------------------------------------
void compute_bit_positions(basct::span<unsigned>& positions, basct::cspan<uint8_t> blob) noexcept;
} // namespace sxt::basbt
