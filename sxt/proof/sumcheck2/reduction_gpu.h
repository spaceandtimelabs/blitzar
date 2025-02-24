#pragma once

#include "sxt/algorithm/base/identity_mapper.h"
#include "sxt/algorithm/reduction/kernel_fit.h"
#include "sxt/algorithm/reduction/thread_reduction.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/field/accumulator.h"
#include "sxt/base/field/element.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/execution/kernel/kernel_dims.h"
#include "sxt/execution/kernel/launch.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"

namespace sxt::prfsk2 {
//--------------------------------------------------------------------------------------------------
// reduction_kernel
//--------------------------------------------------------------------------------------------------
template <unsigned BlockSize, basfld::element T>
__global__ static void reduction_kernel(T* __restrict__ out,
                                        const T* __restrict__ partials,
                                        unsigned n) noexcept {
  auto thread_index = threadIdx.x;
  auto block_index = blockIdx.x;
  auto coefficient_index = blockIdx.y;
  auto index = block_index * (BlockSize * 2) + thread_index;
  auto step = BlockSize * 2 * gridDim.x;
  __shared__ T shared_data[2 * BlockSize];

  // coefficient adjustment
  out += coefficient_index;
  partials += coefficient_index * n;

  // mapper
  algb::identity_mapper<T> mapper{partials};

  // reduce
  algr::thread_reduce<basfld::accumulator<T>, BlockSize>(out + block_index, shared_data, mapper, n,
                                                         step, thread_index, index);
}

//--------------------------------------------------------------------------------------------------
// reduce_sums
//--------------------------------------------------------------------------------------------------
template <basfld::element T>
xena::future<> reduce_sums(basct::span<T> p, basdv::stream& stream,
                           basct::cspan<T> partial_terms) noexcept {
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
  memmg::managed_array<T> p_dev{num_coefficients * dims.num_blocks, &resource};

  // launch kernel
  xenk::launch_kernel(dims.block_size, [&]<unsigned BlockSize>(
                                           std::integral_constant<unsigned, BlockSize>) noexcept {
    reduction_kernel<BlockSize>
        <<<dim3(dims.num_blocks, num_coefficients, 1), BlockSize, 0, stream>>>(
            p_dev.data(), partial_terms.data(), n);
  });

  // copy polynomial to host
  memmg::managed_array<T> p_host_data;
  basct::span<T> p_host = p;
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
      add(p[coefficient_index], p[coefficient_index],
          p_host[coefficient_index * dims.num_blocks + block_index]);
    }
  }
}
} // namespace sxt::prfsk2
