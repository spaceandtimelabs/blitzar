/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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
#include "sxt/algorithm/reduction/kernel_fit.h"

#include <algorithm>

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
  if (n < 4) {
    return {
        .num_blocks = 1,
        .block_size = xenk::block_size_t::v1,
    };
  }
  if (n < 8) {
    return {
        .num_blocks = 1,
        .block_size = xenk::block_size_t::v2,
    };
  }
  if (n < 16) {
    return {
        .num_blocks = 1,
        .block_size = xenk::block_size_t::v4,
    };
  }
  if (n < 32) {
    return {
        .num_blocks = 1,
        .block_size = xenk::block_size_t::v8,
    };
  }
  if (n < 64) {
    return {
        .num_blocks = 1,
        .block_size = xenk::block_size_t::v16,
    };
  }

  // first increase blocks up to 64
  if (n < 64 * 64) {
    return {
        // note: guided by benchmarks, we use 16 instead of 64 here
        // however, this should be reassessed with the production GPU
        .num_blocks = std::min(16u, n / 64),
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
        // note: guided by benchmarks, we use 16 instead of 64 here
        // however, this should be reassessed with the production GPU
        .num_blocks = 16,
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
        // note: guided by benchmarks, we use 16 instead of 64 here
        // however, this should be reassessed with the production GPU
        .num_blocks = 16,
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
        // note: guided by benchmarks, we use 32 instead of 64 here
        // however, this should be reassessed with the production GPU
        .num_blocks = 32,
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
        // note: guided by benchmarks, we use 32 instead of 96 here
        // however, this should be reassessed with the production GPU
        .num_blocks = 32,
        .block_size = xenk::block_size_t::v128,
    };
  }

  return {
      .num_blocks = 128,
      .block_size = xenk::block_size_t::v128,
  };
}
} // namespace sxt::algr
