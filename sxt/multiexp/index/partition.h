#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::mtxi {
//--------------------------------------------------------------------------------------------------
// partition_row
//--------------------------------------------------------------------------------------------------
void partition_row(basct::span<uint64_t>& indexes,
                   uint64_t partition_size) noexcept;

//--------------------------------------------------------------------------------------------------
// partition_rows
//--------------------------------------------------------------------------------------------------
size_t partition_rows(basct::span<basct::span<uint64_t>> rows,
                     uint64_t partition_size) noexcept;
}  // namespace sxt::mtxi
