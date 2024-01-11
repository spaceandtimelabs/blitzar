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
                          basct::cspan<uint8_t*> scalars, unsigned n, unsigned element_num_bytes,
                          unsigned bit_width, unsigned num_partitions) noexcept;
} // namespace sxt::mtxbk
