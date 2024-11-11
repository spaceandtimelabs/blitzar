#include "sxt/proof/sumcheck/reduction_gpu.h"

#include "sxt/algorithm/reduction/kernel_fit.h"
#include "sxt/algorithm/reduction/thread_reduction.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/kernel/kernel_dims.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// reduction_kernel 
//--------------------------------------------------------------------------------------------------
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
static void reduction_kernel(s25t::element* __restrict__ out,
                             const s25t::element* __restrict__ partials, unsigned n) noexcept {}
#pragma clang diagnostic pop

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
  auto dims = algr::fit_reduction_kernel(n);

  // p_dev
  memr::async_device_resource resource{stream};
  memmg::managed_array<s25t::element> p_dev{num_coefficients * dims.num_blocks, &resource};

  // launch kernel
  (void)reduction_kernel;

  // complete reduction on host if necessary
  return {};
}
#pragma clang diagnostic pop
} // namespace sxt::prfsk
