#include "sxt/proof/sumcheck/reduction_gpu.h"

#include "sxt/algorithm/base/identity_mapper.h"
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
#include "sxt/scalar25/operation/accumulator.h"
#include "sxt/scalar25/operation/add.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// reduction_kernel 
//--------------------------------------------------------------------------------------------------
template <unsigned BlockSize>
__global__ static void reduction_kernel(s25t::element* __restrict__ out,
                                        const s25t::element* __restrict__ partials,
                                        unsigned n) noexcept {
  auto thread_index = threadIdx.x;
  auto block_index = blockIdx.x;
  auto coefficient_index = blockIdx.y;
  auto index = block_index * (BlockSize * 2) + thread_index;
  auto step = BlockSize * 2 * gridDim.x;
  __shared__ s25t::element shared_data[2 * BlockSize];

  // coefficient adjustment
  out += coefficient_index;
  partials += coefficient_index * n;

  // mapper
  algb::identity_mapper<s25t::element> mapper{partials};

  // reduce
  algr::thread_reduce<s25o::accumulator, BlockSize>(out + block_index, shared_data, mapper, n, step,
                                                    thread_index, index);
}

//--------------------------------------------------------------------------------------------------
// reduce_sums 
//--------------------------------------------------------------------------------------------------
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
  basct::span<s25t::element> p_host = p;
  if (dims.num_blocks > 1) {
    p_host_data.resize(p_dev.size());
    p_host = p_host_data;
  }
  basdv::async_copy_device_to_host(p_host, p_dev, stream);
  co_await xendv::await_stream(stream);

  // complete reduction on host if necessary
  if (dims.num_blocks == 1) {
    co_return;
  }
  for (unsigned coefficient_index = 0; coefficient_index < num_coefficients; ++coefficient_index) {
    p[coefficient_index] = p_host[coefficient_index * dims.num_blocks];
    for (unsigned block_index = 1; block_index < dims.num_blocks; ++block_index) {
      s25o::add(p[coefficient_index], p[coefficient_index],
                p_host[coefficient_index * dims.num_blocks + block_index]);
    }
  }
}
} // namespace sxt::prfsk