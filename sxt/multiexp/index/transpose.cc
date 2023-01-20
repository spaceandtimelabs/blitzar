#include "sxt/multiexp/index/transpose.h"

#include <vector>

#include "sxt/base/error/assert.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/index/index_table_utility.h"

namespace sxt::mtxi {
//--------------------------------------------------------------------------------------------------
// transpose
//--------------------------------------------------------------------------------------------------
size_t transpose(index_table& table, basct::cspan<basct::cspan<uint64_t>> rows,
                 size_t distinct_entry_count, size_t padding,
                 basf::function_ref<size_t(basct::cspan<uint64_t>)> offset_functor) noexcept {
  // count entries
  std::vector<size_t> counts(distinct_entry_count, padding);
  size_t num_entries = distinct_entry_count * padding;
  for (auto row : rows) {
    for (auto x : row.subspan(offset_functor(row))) {
      SXT_DEBUG_ASSERT(x < distinct_entry_count);
      ++counts[x];
      ++num_entries;
    }
  }
  table.reshape(distinct_entry_count, num_entries);

  // reserve space
  init_rows(table, counts);

  // add padding
  if (padding > 0) {
    for (auto& row : table.header()) {
      std::fill_n(row.data(), padding, 0);
      row = {row.data(), padding};
    }
  }

  // populate
  for (size_t row_index = 0; row_index < rows.size(); ++row_index) {
    auto row = rows[row_index];
    for (auto x : row.subspan(offset_functor(row))) {
      auto& row_t = table.header()[x];
      row_t[row_t.size()] = row_index;
      row_t = {row_t.data(), row_t.size() + 1};
    }
  }

  return num_entries - distinct_entry_count * padding - rows.size();
}

size_t transpose(index_table& table, basct::cspan<basct::cspan<uint64_t>> rows,
                 size_t distinct_entry_count, size_t padding) noexcept {
  return transpose(table, rows, distinct_entry_count, padding,
                   [](basct::cspan<uint64_t> /*row*/) noexcept { return 0; });
}
} // namespace sxt::mtxi
