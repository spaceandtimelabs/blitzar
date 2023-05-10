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
#include "sxt/multiexp/pippenger_multiprod/clump_inputs.h"

#include "sxt/base/container/span.h"
#include "sxt/base/container/span_void.h"
#include "sxt/multiexp/index/clump2_descriptor.h"
#include "sxt/multiexp/index/clump2_descriptor_utility.h"
#include "sxt/multiexp/index/clump2_marker_utility.h"
#include "sxt/multiexp/index/marker_transformation.h"
#include "sxt/multiexp/index/reindex.h"
#include "sxt/multiexp/pippenger_multiprod/active_offset.h"
#include "sxt/multiexp/pippenger_multiprod/driver.h"
#include "sxt/multiexp/pippenger_multiprod/prune.h"
#include "sxt/multiexp/pippenger_multiprod/reduction_stats.h"

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// clump_inputs
//--------------------------------------------------------------------------------------------------
void clump_inputs(basct::span_void inout, reduction_stats& stats,
                  basct::span<basct::span<uint64_t>> products, size_t& num_inactive_outputs,
                  size_t& num_inactive_inputs, const driver& drv, size_t clump_size) noexcept {
  mtxi::clump2_descriptor clump2_descriptor;
  mtxi::init_clump2_descriptor(clump2_descriptor, clump_size);
  auto num_entries_p = mtxi::apply_marker_transformation(
      products.subspan(num_inactive_outputs),
      [clump2_descriptor](basct::span<uint64_t>& indexes) noexcept {
        return mtxi::consume_clump2_marker(indexes, clump2_descriptor);
      },
      compute_active_offset);
  stats.prev_num_terms = num_entries_p;
  std::vector<uint64_t> markers(num_entries_p);
  basct::span<uint64_t> markers_view{markers};
  mtxi::reindex_rows(products.subspan(num_inactive_outputs), markers_view, compute_active_offset);
  stats.num_terms = markers_view.size();
  markers.resize(markers_view.size());
  prune_rows(products, markers, num_inactive_outputs, num_inactive_inputs);
  drv.apply_clump2_operation(inout, markers, clump2_descriptor);
}
} // namespace sxt::mtxpmp
