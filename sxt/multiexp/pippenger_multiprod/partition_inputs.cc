#include "sxt/multiexp/pippenger_multiprod/partition_inputs.h"

#include "sxt/base/container/span.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/index/partition.h"
#include "sxt/multiexp/index/reindex.h"
#include "sxt/multiexp/pippenger_multiprod/driver.h"

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// partition_inputs
//--------------------------------------------------------------------------------------------------
void partition_inputs(memmg::managed_array<void>& inputs,
                      mtxi::index_table& products, const driver& drv,
                      size_t partition_size) noexcept {
  auto num_entries_p = mtxi::partition_rows(products.header(), partition_size);

  memmg::managed_array<uint64_t> partition_markers_data(num_entries_p);
  basct::span<uint64_t> partition_markers{partition_markers_data.data(),
                                          num_entries_p};
  mtxi::reindex_rows(products.header(), partition_markers);

  drv.apply_partition_operation(inputs, partition_markers, partition_size);
}
} // namespace sxt::mtxpmp
