#include "sxt/proof/sumcheck/reduction_gpu.h"

#include "sxt/algorithm/reduction/kernel_fit.h"
#include "sxt/algorithm/reduction/thread_reduction.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/execution/kernel/kernel_dims.h"
#include "sxt/execution/kernel/launch.h"
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
template <unsigned BlockSize>
__global__ static void reduction_kernel(s25t::element* __restrict__ out,
                                        const s25t::element* __restrict__ partials,
                                        unsigned n) noexcept {}
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
  xenk::launch_kernel(dims.block_size, [&]<unsigned BlockSize>(
                                           std::integral_constant<unsigned, BlockSize>) noexcept {
    reduction_kernel<BlockSize>
        <<<dim3(dims.num_blocks, num_coefficients, 1), BlockSize, 0, stream>>>(
            p_dev.data(), partial_terms.data(), n);
  });

  // copy polynomial to host
  memmg::managed_array<s25t::element> p_host_data;
  basct::span<s25t::element> p_host = p_dev;
  if (dims.num_blocks > 1) {
    p_host_data.resize(p_dev.size());
    p_host = p_host_data;
  }
  basdv::async_copy_device_to_host(p_host, p_dev, stream);
  co_await xendv::await_stream(stream);

  // complete reduction on host if necessary
}
#pragma clang diagnostic pop
} // namespace sxt::prfsk