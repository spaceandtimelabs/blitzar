#pragma once

namespace sxt::algb {
//--------------------------------------------------------------------------------------------------
// reducer
//--------------------------------------------------------------------------------------------------
/**
 * Describe a generic reduction function that can be used in CUDA kernels.
 *
 * Note: we require support for volatile references to support efficient reductions
 * within a warp. With volatile, we can avoid the overhead of calling __syncthreads().
 *
 * See https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf, page 22
 */
template <class R>
concept reducer = requires(typename R::value_type& x, volatile typename R::value_type& xv,
                           volatile typename R::value_type& yv, typename R::value_type& z) {
  { R::accumulate_inplace(x, z) } noexcept;
  { R::accumulate_inplace(xv, yv) } noexcept;
};
} // namespace sxt::algb
