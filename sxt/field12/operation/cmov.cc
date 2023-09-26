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
#include "sxt/field12/operation/cmov.h"

#include "sxt/field12/type/element.h"

namespace sxt::f12o {
//--------------------------------------------------------------------------------------------------
// cmov
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE void cmov(f12t::element& f, const f12t::element& g, unsigned int b) noexcept {
  const uint64_t mask = static_cast<uint64_t>(-static_cast<uint64_t>(b));
  for (size_t i = 0; i < g.num_limbs_v; ++i) {
    f[i] = f[i] ^ (mask & (f[i] ^ g[i]));
  }
}

//--------------------------------------------------------------------------------------------------
// cmov
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE void cmov(uint8_t& f, const uint8_t g, unsigned int b) noexcept {
  // If choice = 0, mask = (-0) = 0000...0000
  // If choice = 1, mask = (-1) = 1111...1111
  const uint8_t mask = static_cast<uint8_t>(-static_cast<uint8_t>(b));
  f = f ^ (mask & (f ^ g));
}
} // namespace sxt::f12o
