#pragma once

#include <cassert>

#include "sxt/algorithm/iteration/kernel_fit.h"
#include "sxt/base/container/span.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/property.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/field/element.h"
#include "sxt/base/iterator/split.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/execution/kernel/kernel_dims.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/proof/sumcheck/mle_utility.h"
#include "sxt/scalar25/operation/mul.h"
#include "sxt/scalar25/operation/muladd.h"
#include "sxt/scalar25/operation/sub.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

namespace sxt::prfsk2 {
//--------------------------------------------------------------------------------------------------
// fold_kernel
//--------------------------------------------------------------------------------------------------
template <basfld::element T>
__global__ void fold_kernel(T* __restrict__ mles, unsigned np, unsigned split, T r,
                            T one_m_r) noexcept {
  auto thread_index = threadIdx.x;
  auto block_index = blockIdx.x;
  auto block_size = blockDim.x;
  auto k = basn::divide_up(split, gridDim.x * block_size) * block_size;
  auto block_first = block_index * k;
  assert(block_first < split && "every block should be active");
  auto m = umin(block_first + k, split);

  // adjust mles
  mles += np * blockIdx.y;

  // fold
  auto index = block_first + thread_index;
  for (; index < m; index += block_size) {
    auto x = mles[index];
    mul(x, x, one_m_r);
    auto index_p = split + index;
    if (index_p < np) {
      muladd(x, mles[index_p], r, x);
    }
    mles[index] = x;
  }
}

//--------------------------------------------------------------------------------------------------
// fold_gpu
//--------------------------------------------------------------------------------------------------
/* xena::future<> fold_gpu(basct::span<s25t::element> mles_p, */
/*                         const basit::split_options& split_options, basct::cspan<s25t::element> mles, */
/*                         unsigned n, const s25t::element& r) noexcept; */
/*  */
/* xena::future<> fold_gpu(basct::span<s25t::element> mles_p, basct::cspan<s25t::element> mles, */
/*                         unsigned n, const s25t::element& r) noexcept; */
} // namespace sxt::prfsk2
