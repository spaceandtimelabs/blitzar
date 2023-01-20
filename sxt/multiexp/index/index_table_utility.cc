#include "sxt/multiexp/index/index_table_utility.h"

#include "sxt/base/error/assert.h"
#include "sxt/multiexp/index/index_table.h"

namespace sxt::mtxi {
//--------------------------------------------------------------------------------------------------
// init_rows
//--------------------------------------------------------------------------------------------------
void init_rows(index_table& table, basct::cspan<size_t> sizes) noexcept {
  SXT_DEBUG_ASSERT(table.num_rows() == sizes.size());
  auto entry_data = table.entry_data();
  auto rows = table.header();
  for (size_t row_index = 0; row_index < sizes.size(); ++row_index) {
    rows[row_index] = {entry_data, 0};
    entry_data += sizes[row_index];
  }
}
} // namespace sxt::mtxi
