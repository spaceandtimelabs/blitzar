#include "sxt/algorithm/reduction/kernel_fit.h"

#include "sxt/base/error/assert.h"
#include "sxt/execution/kernel/kernel_dims.h"

namespace sxt::algr {
//--------------------------------------------------------------------------------------------------
// fit_reduction_kernel
//--------------------------------------------------------------------------------------------------
/**
 * Determine parameters for a reduction kernel.
 *
 * Following guidance from
 *  https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf, pages 29-31
 * we choose the degree of parallelism O(n / log_2 n)
 *
 * Note: for now, this is just picking some reasonable values to start with. The choices here
 * are not informed by experimentation and benchmarking yet.
 */
xenk::kernel_dims fit_reduction_kernel(unsigned int n) noexcept {
  SXT_DEBUG_ASSERT(n > 0);
  if (n < 64) {
    // run on host
    return {};
  }

  // first increase blocks up to 64
  if (n < 64 * 64) {
    return {
        .num_blocks = n / 64,
        .block_size = xenk::block_size_t::v32,
    };
  }

  // Fix num blocks at 64 and increase iterations up until we reach a number that would
  // align with Brent's theorem for a block_size increase
  //
  // 64 * 64 * log_2(n) == n when n = 65'536
  // 2 x 65'536 = 131,072
  if (n < 131'073) {
    return {
        .num_blocks = 64,
        .block_size = xenk::block_size_t::v32,
    };
  }

  // Set block size to 64 and increase iterations until we reach a number
  // that aligns with Brent's algorithm for a block increase of 32
  //
  // 96 * 64 * log_2(n) == n when n = 102'247
  // 2 x 102'247 = 204494
  if (n < 204'495) {
    return {
        .num_blocks = 64,
        .block_size = xenk::block_size_t::v64,
    };
  }

  // Set num blocks to 96 and increase iterations until we reach a number
  // that aligns with Brent's algorithm for a block_size increase of 64
  //
  // 96 * 128 * log_2(n) == n when n = 217'907
  // 4 x 217'907 = 787'276
  if (n < 787277) {
    return {
        .num_blocks = 96,
        .block_size = xenk::block_size_t::v64,
    };
  }

  // Set block size to 128 and increase iterations until we reach a number
  // that aligns with Brent's algorithm for a num blocks increase of 32
  //
  // 128 * 128 * log_2(n) == n when n = 297'937
  // 8 x 927'937 = 2'383'496
  if (n < 2'383'497) {
    return {
        .num_blocks = 96,
        .block_size = xenk::block_size_t::v128,
    };
  }

  return {
      .num_blocks = 128,
      .block_size = xenk::block_size_t::v128,
  };
}
} // namespace sxt::algr
