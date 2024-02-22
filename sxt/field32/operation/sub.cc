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
#include "sxt/field32/operation/sub.h"

#include "sxt/field32/base/reduce.h"

namespace sxt::f32o {
//--------------------------------------------------------------------------------------------------
// sub
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void sub(f32t::element& h, const f32t::element& f, const f32t::element& g) noexcept {
  // Compute a - b as ((a + 2^4 * p) - b) to avoid underflow.
  uint64_t z[10] = {0};

  z[0] = (f[0] + (0x3ffffed << 4)) - g[0];
  z[1] = (f[1] + (0x1ffffff << 4)) - g[1];
  z[2] = (f[2] + (0x3ffffff << 4)) - g[2];
  z[3] = (f[3] + (0x1ffffff << 4)) - g[3];
  z[4] = (f[4] + (0x3ffffff << 4)) - g[4];
  z[5] = (f[5] + (0x1ffffff << 4)) - g[5];
  z[6] = (f[6] + (0x3ffffff << 4)) - g[6];
  z[7] = (f[7] + (0x1ffffff << 4)) - g[7];
  z[8] = (f[8] + (0x3ffffff << 4)) - g[8];
  z[9] = (f[9] + (0x1ffffff << 4)) - g[9];

  f32b::reduce(h.data(), z);
}
} // namespace sxt::f32o
