#pragma once

#include <cstddef>
#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// compute_active_offset
//--------------------------------------------------------------------------------------------------
size_t compute_active_offset(basct::cspan<uint64_t> row) noexcept;
}  // namespace sxt::mtxpmp
