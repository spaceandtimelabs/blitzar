#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"
#include "sxt/execution/async/future_fwd.h"

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// transpose_scalars_to_device 
//--------------------------------------------------------------------------------------------------
xena::future<> transpose_scalars_to_device(basct::span<uint8_t> array,
                                           basct::cspan<const uint8_t*> scalars,
                                           unsigned element_num_bytes, 
                                           unsigned bit_width,
                                           unsigned n) noexcept;
} // namespace sxt::mtxb
