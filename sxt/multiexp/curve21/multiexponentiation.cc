#include "sxt/multiexp/curve21/multiexponentiation.h"

#include "sxt/curve21/type/element_p3.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/curve21/multiexponentiation_cpu_driver.h"
#include "sxt/multiexp/curve21/pippenger_multiproduct_solver.h"
#include "sxt/multiexp/pippenger/multiexponentiation2.h"

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// compute_multiexponentiation
//--------------------------------------------------------------------------------------------------
memmg::managed_array<c21t::element_p3>
compute_multiexponentiation(basct::cspan<c21t::element_p3> generators,
                            basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  pippenger_multiproduct_solver solver;
  multiexponentiation_cpu_driver driver{&solver};
  // Note: the cpu driver is non-blocking so that the future upon return the future is
  // available
  return mtxpi::compute_multiexponentiation(driver,
                                            {static_cast<const void*>(generators.data()),
                                             generators.size(), sizeof(c21t::element_p3)},
                                            exponents)
      .await_result()
      .as_array<c21t::element_p3>();
}
} // namespace sxt::mtxc21
