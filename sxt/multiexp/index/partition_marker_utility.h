#pragma once

#include <cstdint>

#include "sxt/base/bit/count.h"
#include "sxt/base/container/span.h"

namespace sxt::mtxi {
//--------------------------------------------------------------------------------------------------
// consume_partition_marker
//--------------------------------------------------------------------------------------------------
uint64_t consume_partition_marker(basct::span<uint64_t>& indexes,
                                  uint64_t partition_size) noexcept;
}  // namespace sxt::mtxi
