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

#include "cub/cub.cuh"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// scalar32
//--------------------------------------------------------------------------------------------------
namespace {
struct scalar32 {
  uint8_t data[32];
};
} // namespace

//--------------------------------------------------------------------------------------------------
// transpose_kernel
//--------------------------------------------------------------------------------------------------
static __global__ void transpose_kernel(uint8_t* __restrict__ dst, const scalar32* __restrict__ src,
                                        unsigned n) noexcept {
  static constexpr unsigned element_num_bytes = 32u;

  auto byte_index = threadIdx.x;
  auto tile_index = blockIdx.x;
  auto num_tiles = gridDim.x;
  auto n_per_tile =
      basn::divide_up(basn::divide_up(n, element_num_bytes), num_tiles) * element_num_bytes;

  auto first = tile_index * n_per_tile;
  auto m = min(n_per_tile, n - first);

  // adjust pointers
  src += first;
  dst += byte_index * n + first;

  // set up algorithm
  using BlockExchange = cub::BlockExchange<uint8_t, element_num_bytes, element_num_bytes>;
  __shared__ BlockExchange::TempStorage temp_storage;

  // transpose
  scalar32 s;
  unsigned out_first = 0;
  for (unsigned i = byte_index; i < n_per_tile; i += element_num_bytes) {
    if (i < m) {
      s = src[i];
    }
    BlockExchange(temp_storage).StripedToBlocked(s.data);
    __syncthreads();
    for (unsigned j = 0; j < 32u; ++j) {
      auto out_index = out_first + j;
      if (out_index < m) {
        dst[out_index] = s.data[j];
      }
    }
    out_first += element_num_bytes;
    __syncthreads();
  }
}

//--------------------------------------------------------------------------------------------------
// transpose_scalars_to_device_impl
//--------------------------------------------------------------------------------------------------
static xena::future<> transpose_scalars_to_device_impl(basct::span<uint8_t> array_slice,
                                                       basct::cspan<uint8_t> scalars,
                                                       unsigned element_num_bytes,
                                                       unsigned bit_width, unsigned n) noexcept {
  SXT_RELEASE_ASSERT(element_num_bytes == 32 && bit_width == 8,
                     "we only support 32 byte scalars for now");
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  memmg::managed_array<uint8_t> scalars_dev{scalars.size(), &resource};
  basdv::async_copy_host_to_device(scalars_dev, scalars, stream);
  auto num_tiles = std::min(basn::divide_up(n, 32u), 64u);

  transpose_kernel<<<num_tiles, 32, 0, stream>>>(
      array_slice.data(), reinterpret_cast<const scalar32*>(scalars_dev.data()), n);

  return xendv::await_and_own_stream(std::move(stream));
}

//--------------------------------------------------------------------------------------------------
// transpose_scalars_to_device
//--------------------------------------------------------------------------------------------------
xena::future<> transpose_scalars_to_device(basct::span<uint8_t> array,
                                           basct::cspan<const uint8_t*> scalars,
                                           unsigned element_num_bytes, unsigned bit_width,
                                           unsigned n) noexcept {
  auto num_outputs = scalars.size();
  SXT_DEBUG_ASSERT(
      // clang-format off
      array.size() == num_outputs * element_num_bytes * n &&
      basdv::is_active_device_pointer(array.data())
      // clang-format on
  );
  std::vector<xena::future<>> futs(num_outputs);
  for (size_t output_index = 0; output_index < num_outputs; ++output_index) {
    futs[output_index] = transpose_scalars_to_device_impl(
        array.subspan(output_index * element_num_bytes * n, element_num_bytes * n),
        basct::cspan<uint8_t>{scalars[output_index], n * element_num_bytes}, element_num_bytes,
        bit_width, n);
  }
  for (auto& fut : futs) {
    co_await std::move(fut);
  }
}
} // namespace sxt::mtxb
