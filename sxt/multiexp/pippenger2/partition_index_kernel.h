#pragma once

#include <cstdint>

#include "sxt/base/type/raw_stream.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// launch_fill_partition_indexes_kernel 
//--------------------------------------------------------------------------------------------------
void launch_fill_partition_indexes_kernel(uint16_t* __restrict__ indexes, bast::raw_stream_t stream,
                                          const uint8_t* __restrict__ scalars, unsigned num_outputs,
                                          unsigned n);
} // namespace sxt::mtxpp2
