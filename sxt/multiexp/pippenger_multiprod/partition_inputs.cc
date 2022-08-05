#include "sxt/multiexp/pippenger_multiprod/partition_inputs.h"

#include <vector>

#include "sxt/base/container/span.h"
#include "sxt/base/container/span_void.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/index/marker_transformation.h"
#include "sxt/multiexp/index/partition_marker_utility.h"
#include "sxt/multiexp/index/reindex.h"
#include "sxt/multiexp/pippenger_multiprod/active_offset.h"
#include "sxt/multiexp/pippenger_multiprod/driver.h"
#include "sxt/multiexp/pippenger_multiprod/prune.h"
#include "sxt/multiexp/pippenger_multiprod/reduction_stats.h"

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// partition_inputs
//--------------------------------------------------------------------------------------------------
void partition_inputs(basct::span_void inout, reduction_stats& stats,
                      basct::span<basct::span<uint64_t>> products, size_t& num_inactive_outputs,
                      size_t& num_inactive_inputs, const driver& drv,
                      size_t partition_size) noexcept {
  auto num_entries_p = mtxi::apply_marker_transformation(
      products.subspan(num_inactive_outputs),
      [partition_size](basct::span<uint64_t>& indexes) noexcept {
        return mtxi::consume_partition_marker(indexes, partition_size);
      },
      compute_active_offset);
  stats.prev_num_terms = num_entries_p;

  std::vector<uint64_t> markers(num_entries_p);
  basct::span<uint64_t> markers_view{markers.data(), num_entries_p};
  mtxi::reindex_rows(products.subspan(num_inactive_outputs), markers_view, compute_active_offset);
  markers.resize(markers_view.size());
  stats.num_terms = markers.size();

  drv.apply_partition_operation(inout, markers, partition_size);
}
} // namespace sxt::mtxpmp
