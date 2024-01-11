#pragma once

#include <cstddef>

#include "sxt/base/container/span.h"
#include "sxt/execution/async/future_fwd.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// count_bucket_entries
//--------------------------------------------------------------------------------------------------
xena::future<> count_bucket_entries(basct::span<unsigned> count_array,
                                    basct::cspan<uint8_t> scalars, unsigned element_num_bytes,
                                    unsigned bit_width, unsigned num_partitions) noexcept;
} // namespace sxt::mtxbk
