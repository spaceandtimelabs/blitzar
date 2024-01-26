#pragma once

#include <cstddef>

#include "sxt/base/container/span.h"
#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::basdv { class stream; }

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// count_bucket_entries
//--------------------------------------------------------------------------------------------------
void count_bucket_entries(memmg::managed_array<unsigned>& count_array, const basdv::stream& stream,
                          basct::cspan<uint8_t> scalars, unsigned element_num_bytes, unsigned n,
                          unsigned num_outputs, unsigned bit_width,
                          unsigned num_partitions) noexcept;

void count_bucket_entries(basct::span<unsigned> count_array, const basdv::stream& stream,
                          basct::cspan<uint8_t> scalars, unsigned element_num_bytes, unsigned n,
                          unsigned bit_width) noexcept;
} // namespace sxt::mtxbk
