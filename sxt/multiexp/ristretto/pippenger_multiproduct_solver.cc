#include "sxt/multiexp/ristretto/pippenger_multiproduct_solver.h"

#include <cassert>

#include "sxt/base/container/span.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/curve21/multiproduct_cpu_driver.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/pippenger_multiprod/multiproduct.h"

namespace sxt::mtxrs {
//--------------------------------------------------------------------------------------------------
// workspace_size
//--------------------------------------------------------------------------------------------------
size_t pippenger_multiproduct_solver::workspace_size(size_t /*num_multiproduct_inputs*/,
                                                     size_t num_entries) const noexcept {
  return num_entries;
}

//--------------------------------------------------------------------------------------------------
// solve
//--------------------------------------------------------------------------------------------------
void pippenger_multiproduct_solver::solve(memmg::managed_array<void>& inout,
                                          mtxi::index_table& multiproduct_table,
                                          size_t num_inputs) const noexcept {
  if (num_inputs == 0) {
    return;
  }
  mtxc21::multiproduct_cpu_driver driver;
  mtxpmp::compute_multiproduct(inout, multiproduct_table.header(), driver, num_inputs);
}
} // namespace sxt::mtxrs
