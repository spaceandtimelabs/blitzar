#include "sxt/proof/sumcheck/reduction_gpu.h"

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/async/future.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// reduce_sums 
//--------------------------------------------------------------------------------------------------
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
xena::future<> reduce_sums(basct::span<s25t::element> p, basdv::stream& stream,
                           basct::cspan<s25t::element> partial_terms) noexcept {
  auto num_coefficients = p.size();
  auto n = partial_terms.size() / num_coefficients;
  SXT_DEBUG_ASSERT(
      // clang-format off
      n > 0 &&
      partial_terms.size() == num_coefficients * n && 
      basdv::is_host_pointer(p.data()) &&
      basdv::is_active_device_pointer(partial_terms.data())
      // clang-format on
  );
  return {};
}
#pragma clang diagnostic pop
} // namespace sxt::prfsk
