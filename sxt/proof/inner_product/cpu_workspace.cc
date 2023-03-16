#include "sxt/proof/inner_product/cpu_workspace.h"

#include "sxt/execution/async/future.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// ap_value
//--------------------------------------------------------------------------------------------------
xena::future<void> cpu_workspace::ap_value(s25t::element& value) const noexcept {
  value = this->a_vector[0];
  return xena::make_ready_future();
}
} // namespace sxt::prfip
