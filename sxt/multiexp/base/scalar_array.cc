/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
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
#include "sxt/multiexp/base/scalar_array.h"

#include <vector>

/*
 * This is a workaround to define _VSTD before including cub/cub.cuh.
 * It should be removed when we can upgrade to a newer version of CUDA.
 *
 * We need to define _VSTD in order to use the clang version defined in
 * clang.nix and the CUDA toolkit version defined in cuda.nix.
 *
 * _VSTD was deprecated and removed from the LLVM truck.
 * NVIDIA: https://github.com/NVIDIA/cccl/pull/1331
 * LLVM: https://github.com/llvm/llvm-project/commit/683bc94e1637bd9bacc978f5dc3c79cfc8ff94b9
 *
 * We cannot currently use any CUDA toolkit above 12.4.1 because the Kubernetes
 * cluster currently cannot install a driver above 550.
 *
 * See CUDA toolkit and driver support:
 * https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
 */
#include <__config>
#define _VSTD std::_LIBCPP_ABI_NAMESPACE

#include "cub/cub.cuh"
#include "sxt/base/container/span_utility.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/base/num/constexpr_switch.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"

    namespace sxt::mtxb {
  //--------------------------------------------------------------------------------------------------
  // scalar_blob
  //--------------------------------------------------------------------------------------------------
  namespace {
  template <unsigned NumBytes> struct scalar_blob {
    uint8_t data[NumBytes];
  };
  } // namespace

  //--------------------------------------------------------------------------------------------------
  // transpose_kernel
  //--------------------------------------------------------------------------------------------------
  template <unsigned NumBytes>
  static __global__ void transpose_kernel(uint8_t* __restrict__ dst,
                                          const scalar_blob<NumBytes>* __restrict__ src,
                                          unsigned n) noexcept {
    using Scalar = scalar_blob<NumBytes>;

    auto byte_index = threadIdx.x;
    auto tile_index = blockIdx.x;
    auto output_index = blockIdx.y;
    auto num_tiles = gridDim.x;
    auto n_per_tile = basn::divide_up(basn::divide_up(n, NumBytes), num_tiles) * NumBytes;

    auto first = tile_index * n_per_tile;
    auto m = min(n_per_tile, n - first);

    // adjust pointers
    src += first;
    src += output_index * n;
    dst += byte_index * n + first;
    dst += output_index * NumBytes * n;

    // set up algorithm
    using BlockExchange = cub::BlockExchange<uint8_t, NumBytes, NumBytes>;
    __shared__ typename BlockExchange::TempStorage temp_storage;

    // transpose
    Scalar s;
    unsigned out_first = 0;
    for (unsigned i = byte_index; i < n_per_tile; i += NumBytes) {
      if (i < m) {
        s = src[i];
      }
      BlockExchange(temp_storage).StripedToBlocked(s.data);
      __syncthreads();
      for (unsigned j = 0; j < NumBytes; ++j) {
        auto out_index = out_first + j;
        if (out_index < m) {
          dst[out_index] = s.data[j];
        }
      }
      out_first += NumBytes;
      __syncthreads();
    }
  }

  //--------------------------------------------------------------------------------------------------
  // transpose_scalars_to_device
  //--------------------------------------------------------------------------------------------------
  xena::future<> transpose_scalars_to_device(basct::span<uint8_t> array,
                                             basct::cspan<const uint8_t*> scalars,
                                             unsigned element_num_bytes, unsigned n) noexcept {
    auto num_outputs = static_cast<unsigned>(scalars.size());
    if (n == 0 || num_outputs == 0) {
      co_return;
    }
    SXT_DEBUG_ASSERT(
        // clang-format off
      array.size() == num_outputs * element_num_bytes * n &&
      basdv::is_active_device_pointer(array.data()) &&
      basdv::is_host_pointer(scalars[0])
        // clang-format on
    );
    basdv::stream stream;
    memr::async_device_resource resource{stream};
    memmg::managed_array<uint8_t> array_p{array.size(), &resource};
    auto num_bytes_per_output = element_num_bytes * n;
    for (size_t output_index = 0; output_index < num_outputs; ++output_index) {
      basdv::async_copy_host_to_device(
          basct::subspan(array_p, output_index * num_bytes_per_output, num_bytes_per_output),
          basct::cspan<uint8_t>{scalars[output_index], num_bytes_per_output}, stream);
    }
    auto num_tiles = std::min(basn::divide_up(n, num_outputs * element_num_bytes), 64u);
    auto num_bytes_log2 = basn::ceil_log2(element_num_bytes);
    basn::constexpr_switch<6>(
        num_bytes_log2,
        [&]<unsigned LogNumBytes>(std::integral_constant<unsigned, LogNumBytes>) noexcept {
          constexpr auto NumBytes = 1u << LogNumBytes;
          SXT_DEBUG_ASSERT(NumBytes == element_num_bytes);
          transpose_kernel<NumBytes><<<dim3(num_tiles, num_outputs, 1), NumBytes, 0, stream>>>(
              array.data(), reinterpret_cast<scalar_blob<NumBytes>*>(array_p.data()), n);
        });
    co_await xendv::await_stream(std::move(stream));
  }
} // namespace sxt::mtxb
