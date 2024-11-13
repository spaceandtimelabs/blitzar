#include "sxt/proof/sumcheck/sum_gpu.h"

#include "sxt/execution/kernel/launch.h"
#include "sxt/algorithm/reduction/kernel_fit.h"
#include "sxt/base/device/stream.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/kernel/kernel_dims.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/proof/sumcheck/reduction_gpu.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// partial_sum_kernel 
//--------------------------------------------------------------------------------------------------
template <unsigned BlockSize>
__global__ static void
partial_sum_kernel(s25t::element* __restrict__ out,
                   const std::pair<s25t::element, unsigned>* __restrict__ product_table,
                   const unsigned* __restrict__ product_terms, unsigned n) noexcept {}

//--------------------------------------------------------------------------------------------------
// partial_sum 
//--------------------------------------------------------------------------------------------------
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
static xena::future<> partial_sum(basct::span<s25t::element> p, basdv::stream& stream,
                                  basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                                  basct::cspan<unsigned> product_terms, unsigned n) noexcept {
  auto num_coefficients = p.size();
  auto dims = algr::fit_reduction_kernel(n);
  memr::async_device_resource resource{stream};

  // partials
  memmg::managed_array<s25t::element> partials{num_coefficients * dims.num_blocks, &resource};
  xenk::launch_kernel(dims.block_size, [&]<unsigned BlockSize>(
                                           std::integral_constant<unsigned, BlockSize>) noexcept {
    partial_sum_kernel<BlockSize>
        <<<dim3(dims.num_blocks, num_coefficients, 1), BlockSize, 0, stream>>>(
            partials.data(), product_table.data(), product_terms.data(), n);
  });

  // reduce partials
  co_await reduce_sums(p, stream, partials);
}
#pragma clang diagnostic pop

//--------------------------------------------------------------------------------------------------
// sum_gpu 
//--------------------------------------------------------------------------------------------------
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
xena::future<> sum_gpu(basct::span<s25t::element> p, basct::cspan<s25t::element> mles,
                       basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                       basct::cspan<unsigned> product_terms, unsigned n) noexcept {
  (void)partial_sum;
  return {};
}
#pragma clang diagnostic pop
} // namespace sxt::prfsk
