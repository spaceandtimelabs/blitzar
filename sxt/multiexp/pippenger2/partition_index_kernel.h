#pragma once

#include <cstdint>

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// fill_partition_index_kernel 
//--------------------------------------------------------------------------------------------------
__global__ void fill_partition_index_kernel(uint16_t* __restrict__ indexes,
                                            const uint8_t* __restrict__ scalars,
                                            unsigned num_outputs, unsigned n);
} // namespace sxt::mtxpp2
