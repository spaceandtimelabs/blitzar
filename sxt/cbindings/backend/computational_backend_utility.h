#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::cbnbck {
//--------------------------------------------------------------------------------------------------
// make_scalars_span 
//--------------------------------------------------------------------------------------------------
basct::cspan<uint8_t> make_scalars_span(const uint8_t* data,
                                        basct::cspan<unsigned> output_bit_table,
                                        basct::cspan<unsigned> output_lengths) noexcept;
} // namespace sxt::cbnbck
