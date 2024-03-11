#pragma once

#include "sxt/base/container/span.h"
#include "sxt/execution/async/future_fwd.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// fill_partition_indexes 
//--------------------------------------------------------------------------------------------------
xena::future<> fill_partition_indexes(basct::span<uint16_t> indexes, basct::cspan<const uint8_t*> scalars,
                                      unsigned element_num_bytes, unsigned n) noexcept;
} // namespace sxt::mtxpp2
