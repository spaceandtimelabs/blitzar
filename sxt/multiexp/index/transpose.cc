/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
      auto sz = row_t.size();
      row_t = {row_t.data(), sz + 1};
      row_t[sz] = row_index;
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
