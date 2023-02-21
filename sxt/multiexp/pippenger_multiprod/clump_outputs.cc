#include "sxt/multiexp/pippenger_multiprod/clump_outputs.h"

#include <iostream>

#include "sxt/base/container/span.h"
#include "sxt/multiexp/index/clump2_descriptor.h"
#include "sxt/multiexp/index/clump2_descriptor_utility.h"
#include "sxt/multiexp/index/clump2_marker_utility.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/index/index_table_utility.h"
#include "sxt/multiexp/index/marker_transformation.h"
#include "sxt/multiexp/index/reindex.h"
#include "sxt/multiexp/index/transpose.h"
#include "sxt/multiexp/pippenger_multiprod/active_offset.h"

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// compute_clumped_output_table
//--------------------------------------------------------------------------------------------------
bool compute_clumped_output_table(mtxi::index_table& table, std::vector<uint64_t>& output_clumps,
                                  basct::cspan<basct::cspan<uint64_t>> rows,
                                  size_t num_active_inputs, size_t clump_size) noexcept {
  mtxi::index_table table_p;
  auto naive_product_count =
      mtxi::transpose(table_p, rows, num_active_inputs, 0, compute_active_offset);
  mtxi::clump2_descriptor clump2_descriptor;
  mtxi::init_clump2_descriptor(clump2_descriptor, clump_size);
  auto num_entries_p = mtxi::apply_marker_transformation(
      table_p.header(), [clump2_descriptor](basct::span<uint64_t>& indexes) noexcept {
        return mtxi::consume_clump2_marker(indexes, clump2_descriptor);
      });
  output_clumps.resize(num_entries_p);
  basct::span<uint64_t> markers{output_clumps};
  mtxi::reindex_rows(table_p.header(), markers);
  if (naive_product_count == num_entries_p - markers.size()) {
    // we didn't reduce the problem to anything simpler
    return false;
  }
  output_clumps.resize(markers.size());
  mtxi::transpose(table, table_p.cheader(), output_clumps.size(), 2);
  for (size_t row_index = 0; row_index < table.num_rows(); ++row_index) {
    table.header()[row_index][0] = row_index;
  }
  return true;
}

//--------------------------------------------------------------------------------------------------
// rewrite_multiproducts_with_output_clumps
//--------------------------------------------------------------------------------------------------
void rewrite_multiproducts_with_output_clumps(basct::span<basct::span<uint64_t>> rows,
                                              basct::cspan<uint64_t> output_clumps,
                                              size_t clump_size) noexcept {
  mtxi::clump2_descriptor clump2_descriptor;
  mtxi::init_clump2_descriptor(clump2_descriptor, clump_size);

  // clear active entries
  for (auto& row : rows) {
    row = {row.data(), 2 + row[1]};
  }

  // fill table
  for (size_t marker_index = 0; marker_index < output_clumps.size(); ++marker_index) {
    uint64_t clump_index, index1, index2;
    mtxi::unpack_clump2_marker(clump_index, index1, index2, clump2_descriptor,
                               output_clumps[marker_index]);
    auto clump_first = clump_size * clump_index;
    auto& row1 = rows[clump_first + index1];
    auto sz = row1.size();
    row1 = {row1.data(), sz + 1};
    row1[sz] = marker_index;
    if (index1 != index2) {
      auto& row2 = rows[clump_first + index2];
      auto sz = row2.size();
      row2 = {row2.data(), sz + 1};
      row2[sz] = marker_index;
    }
  }
}
} // namespace sxt::mtxpmp
