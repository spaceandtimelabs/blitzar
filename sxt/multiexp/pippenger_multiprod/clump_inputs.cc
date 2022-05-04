#include "sxt/multiexp/pippenger_multiprod/clump_inputs.h"

#include "sxt/base/container/span.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/index/clump2_descriptor.h"
#include "sxt/multiexp/index/clump2_descriptor_utility.h"
#include "sxt/multiexp/index/clump2_marker_utility.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/index/marker_transformation.h"
#include "sxt/multiexp/index/reindex.h"
#include "sxt/multiexp/pippenger_multiprod/driver.h"

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// clump_inputs
//--------------------------------------------------------------------------------------------------
void clump_inputs(memmg::managed_array<void>& inputs,
                  mtxi::index_table& products, const driver& drv,
                  size_t clump_size) noexcept {
  mtxi::clump2_descriptor clump2_descriptor;
  mtxi::init_clump2_descriptor(clump2_descriptor, clump_size);
  auto num_entries_p = mtxi::apply_marker_transformation(
      products.header(),
      [clump2_descriptor](basct::span<uint64_t>& indexes) noexcept {
        return mtxi::consume_clump2_marker(indexes, clump2_descriptor);
      });
  memmg::managed_array<uint64_t> markers_data(num_entries_p);
  basct::span<uint64_t> markers{markers_data.data(), num_entries_p};
  mtxi::reindex_rows(products.header(), markers);
  drv.apply_clump2_operation(inputs, markers, clump2_descriptor);
}
} // namespace sxt::mtxpmp
