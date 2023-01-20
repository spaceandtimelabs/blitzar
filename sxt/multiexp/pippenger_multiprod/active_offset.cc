#include "sxt/multiexp/pippenger_multiprod/active_offset.h"

#include "sxt/base/error/assert.h"

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// compute_active_offset
//--------------------------------------------------------------------------------------------------
size_t compute_active_offset(basct::cspan<uint64_t> row) noexcept {
  SXT_DEBUG_ASSERT(row.size() >= 2 && row.size() >= 2 + row[1]);
  return 2 + row[1];
}
} // namespace sxt::mtxpmp
