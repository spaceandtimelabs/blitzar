/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cassert>

#include "sxt/algorithm/base/index_functor.h"
#include "sxt/algorithm/iteration/kernel_fit.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/iterator/chunk_options.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/synchronization.h"
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
  return xendv::await_and_own_stream(std::move(stream));
}

template <algb::index_functor F> xena::future<> for_each(F f, unsigned n) noexcept {
  return for_each(basdv::stream{}, f, n);
}
} // namespace sxt::algi
