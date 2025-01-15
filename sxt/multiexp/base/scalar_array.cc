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

#include <strings.h>

#include <vector>

#include "sxt/base/container/span_utility.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/generate.h"
#include "sxt/execution/device/synchronization.h"

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// transpose_scalars
//--------------------------------------------------------------------------------------------------
void transpose_scalars(basct::span<uint8_t> array, const uint8_t* scalars,
                       unsigned element_num_bytes, unsigned n, size_t offset) noexcept {
  auto remaining_size = array.size();
  auto byte_index = offset / n;

  // copy for first byte position
  auto byte_offset = static_cast<unsigned>(offset - byte_index * n);
  auto chunk_size = std::min(static_cast<size_t>(n - byte_offset), remaining_size);
  auto out = array.data();
  for (unsigned i = 0; i < chunk_size; ++i) {
    *out++ = *(scalars + byte_index + i * element_num_bytes);
  }
  remaining_size -= chunk_size;

  // copy remaining byte positions
  while (remaining_size > 0) {
    ++byte_index;
    auto chunk_size = std::min(static_cast<size_t>(n), remaining_size);
    for (unsigned i = 0; i < chunk_size; ++i) {
      *out++ = *(scalars + byte_index + i * element_num_bytes);
    }
    remaining_size -= chunk_size;
  }
}

//--------------------------------------------------------------------------------------------------
// transpose_scalars_to_device
//--------------------------------------------------------------------------------------------------
xena::future<> transpose_scalars_to_device(basct::span<uint8_t> array,
                                           basct::cspan<const uint8_t*> scalars,
                                           unsigned element_num_bytes, unsigned n) noexcept {
  auto num_outputs = static_cast<unsigned>(scalars.size());
  size_t bytes_per_output = element_num_bytes * n;
  if (n == 0 || num_outputs == 0) {
    co_return;
  }
  SXT_DEBUG_ASSERT(
      // clang-format off
      array.size() == num_outputs * bytes_per_output &&
      basdv::is_active_device_pointer(array.data()) &&
      basdv::is_host_pointer(scalars[0])
      // clang-format on
  );
  auto f = [&](basct::span<uint8_t> buffer, size_t index) noexcept {
    auto remaining_bytes = buffer.size();
    auto output_index = index / bytes_per_output;

    // first output
    auto offset = index - output_index * bytes_per_output;
    auto chunk_size = std::min(bytes_per_output - offset, remaining_bytes);
    transpose_scalars(buffer.subspan(0, chunk_size), scalars[output_index], element_num_bytes, n,
                      offset);
    remaining_bytes -= chunk_size;

    // remaining outputs
    while (remaining_bytes > 0) {
      ++output_index;
      chunk_size = std::min(bytes_per_output, remaining_bytes);
      transpose_scalars(buffer.subspan(buffer.size() - remaining_bytes, chunk_size),
                        scalars[output_index], element_num_bytes, n, 0);
      remaining_bytes -= chunk_size;
    }
  };
  basdv::stream stream;
  co_await xendv::generate_to_device(array.subspan(0, num_outputs * bytes_per_output), stream, f);
}
} // namespace sxt::mtxb
