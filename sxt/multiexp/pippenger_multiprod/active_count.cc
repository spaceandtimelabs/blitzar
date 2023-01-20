#include "sxt/multiexp/pippenger_multiprod/active_count.h"

#include "sxt/base/error/assert.h"

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// count_active_entries
//--------------------------------------------------------------------------------------------------
void count_active_entries(basct::span<size_t> counts,
                          basct::cspan<basct::cspan<uint64_t>> rows) noexcept {
  for (auto row : rows) {
    SXT_DEBUG_ASSERT(row.size() >= 2 && row.size() >= 2 + row[1]);
    for (size_t active_index = 2 + row[1]; active_index < row.size(); ++active_index) {
      ++counts[row[active_index]];
    }
  }
}
} // namespace sxt::mtxpmp
