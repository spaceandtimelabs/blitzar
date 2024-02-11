#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::basdv { class stream; }

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// inclusive_prefix_count_buckets 
//--------------------------------------------------------------------------------------------------
void inclusive_prefix_count_buckets(basct::span<unsigned> counts, const basdv::stream& stream,
                                    basct::cspan<uint8_t> digits, unsigned element_num_bytes,
                                    unsigned bit_width, unsigned num_outputs, unsigned tile_size,
                                    unsigned n) noexcept;

void inclusive_prefix_count_buckets(basct::span<unsigned> counts, basct::span<uint16_t> tile_counts,
                                    const basdv::stream& stream, basct::cspan<uint8_t> digits,
                                    unsigned element_num_bytes, unsigned bit_width,
                                    unsigned num_outputs, unsigned tile_size, unsigned n) noexcept;
} // namespace sxt::mtxbk
