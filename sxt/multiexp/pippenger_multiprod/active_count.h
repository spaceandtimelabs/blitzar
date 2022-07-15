#pragma once

#include <cstddef>
#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// count_active_entries
//--------------------------------------------------------------------------------------------------
void count_active_entries(basct::span<size_t> counts,
                          basct::cspan<basct::cspan<uint64_t>> rows) noexcept;

inline void count_active_entries(basct::span<size_t> counts,
                                 basct::cspan<basct::span<uint64_t>> rows) noexcept {
  count_active_entries(counts,
                       {reinterpret_cast<const basct::cspan<uint64_t>*>(rows.data()), rows.size()});
}
} // namespace sxt::mtxpmp
