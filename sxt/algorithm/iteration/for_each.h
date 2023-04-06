#pragma once

#include <cassert>

#include "sxt/algorithm/base/index_functor.h"
#include "sxt/algorithm/iteration/kernel_fit.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/async/synchronization.h"
#include "sxt/execution/kernel/kernel_dims.h"

namespace sxt::algi {
//--------------------------------------------------------------------------------------------------
// for_each_kernel
//--------------------------------------------------------------------------------------------------
template <algb::index_functor F> __global__ void for_each_kernel(F f, unsigned n) {
  auto thread_index = threadIdx.x;
  auto block_index = blockIdx.x;
  auto block_size = blockDim.x;
  auto k = basn::divide_up(n, gridDim.x * block_size) * block_size;
  auto block_first = block_index * k;
  assert(block_first < n && "every block should be active");
  auto m = umin(block_first + k, n);
  auto index = block_first + thread_index;
  for (; index < m; index += block_size) {
    f(n, index);
  }
}

//--------------------------------------------------------------------------------------------------
// launch_for_each_kernel
//--------------------------------------------------------------------------------------------------
template <algb::index_functor F>
void launch_for_each_kernel(bast::raw_stream_t stream, F f, unsigned n) noexcept {
  auto dims = fit_iteration_kernel(n);
  for_each_kernel<<<dims.num_blocks, static_cast<unsigned>(dims.block_size), 0, stream>>>(f, n);
}

//--------------------------------------------------------------------------------------------------
// for_each
//--------------------------------------------------------------------------------------------------
template <algb::index_functor F>
xena::future<> for_each(basdv::stream&& stream, F f, unsigned n) noexcept {
  launch_for_each_kernel(stream, f, n);
  return xena::await_and_own_stream(std::move(stream));
}

template <algb::index_functor F> xena::future<> for_each(F f, unsigned n) noexcept {
  return for_each(basdv::stream{}, f, n);
}
} // namespace sxt::algi
