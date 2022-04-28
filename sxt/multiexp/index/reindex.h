#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::mtxi {
//--------------------------------------------------------------------------------------------------
// reindex_rows
//--------------------------------------------------------------------------------------------------
void reindex_rows(basct::span<basct::span<uint64_t>> rows,
                  basct::span<uint64_t>& values) noexcept;
}  // namespace sxt::mtxi
