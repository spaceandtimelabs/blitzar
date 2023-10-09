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

#include <cstdint>
#include <cstring>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::basbt {
//--------------------------------------------------------------------------------------------------
// store64_le
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
inline void store64_le(uint8_t dst[8], uint64_t w) noexcept {
  // note: assume the architecture is little endian
  std::memcpy(static_cast<void*>(dst), static_cast<const void*>(&w), sizeof(w));
}

//--------------------------------------------------------------------------------------------------
// store64_be
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
inline void store64_be(uint8_t dst[8], uint64_t w) noexcept {
  // note: assume the architecture is little endian
  uint64_t w_le{__builtin_bswap64(w)};
  std::memcpy(static_cast<void*>(dst), static_cast<const void*>(&w_le), sizeof(w_le));
}
} // namespace sxt::basbt
