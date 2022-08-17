#pragma once

#include <cstddef>
#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// compute_radix_log2
//--------------------------------------------------------------------------------------------------
size_t compute_radix_log2(basct::cspan<uint8_t> max_exponent, size_t num_inputs,
                          size_t num_outputs) noexcept;
} // namespace sxt::mtxpi
