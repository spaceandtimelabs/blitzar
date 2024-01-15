#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::basdv { class stream; }

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// make_device_scalar_array 
//--------------------------------------------------------------------------------------------------
void make_device_scalar_array(basct::span<uint8_t> array, const basdv::stream& stream,
                              basct::cspan<const uint8_t*> scalars, size_t element_num_bytes,
                              size_t n) noexcept;
} // namespace sxt::mtxb
