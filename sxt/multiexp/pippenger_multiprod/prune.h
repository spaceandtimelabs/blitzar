#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "sxt/base/container/span.h"

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// prune_rows
//--------------------------------------------------------------------------------------------------
void prune_rows(basct::span<basct::span<uint64_t>> rows,
                    std::vector<uint64_t>& markers, size_t& num_inactive_outputs,
                    size_t& num_inactive_inputs) noexcept;
}  // namespace sxt::mtxpmp
