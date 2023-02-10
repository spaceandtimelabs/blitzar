#include "sxt/multiexp/curve21/pippenger_multiproduct_solver.h"

#include <algorithm>
#include <cstdint>

#include "sxt/base/container/blob_array.h"
#include "sxt/base/error/assert.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/base/generator_utility.h"
#include "sxt/multiexp/curve21/multiproduct_cpu_driver.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/index/reindex.h"
#include "sxt/multiexp/pippenger_multiprod/active_offset.h"
#include "sxt/multiexp/pippenger_multiprod/multiproduct.h"

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// solve
//--------------------------------------------------------------------------------------------------
xena::future<memmg::managed_array<c21t::element_p3>> pippenger_multiproduct_solver::solve(
    mtxi::index_table&& multiproduct_table, basct::cspan<c21t::element_p3> generators,
    const basct::blob_array& masks, size_t num_inputs) const noexcept {
  size_t entry_count = 0;
  for (auto row : multiproduct_table.cheader()) {
    SXT_DEBUG_ASSERT(row.size() > 2, "all outputs should have at least a single product");
    entry_count += row.size() - 2;
  }
  SXT_DEBUG_ASSERT(entry_count >= num_inputs);
  memmg::managed_array<c21t::element_p3> res(entry_count);
  mtxb::filter_generators<c21t::element_p3>(basct::span<c21t::element_p3>{res.data(), num_inputs},
                                            generators, masks);
  mtxc21::multiproduct_cpu_driver driver;
  mtxpmp::compute_multiproduct(res, multiproduct_table.header(), driver, num_inputs);
  return xena::make_ready_future(std::move(res));
}
} // namespace sxt::mtxc21
