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
#include "sxt/field32/base/reduce.h"

namespace sxt::f32b {
//--------------------------------------------------------------------------------------------------
// reduce
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void reduce(uint32_t h[10], const uint64_t f[10]) noexcept {
  constexpr uint64_t LOW_25_BITS = (1 << 25) - 1;
  constexpr uint64_t LOW_26_BITS = (1 << 26) - 1;

  uint64_t z[10];
  for (size_t i = 0; i < 10; ++i) {
    z[i] = f[i];
  }

  // Carry the value from limb i = 0..8 to limb i+1
  auto carry = [](uint64_t z[10], size_t i) {
    if (i % 2 == 0) {
      // Even limbs have 26 bits
      z[i + 1] += z[i] >> 26;
      z[i] &= LOW_26_BITS;
    } else {
      // Odd limbs have 25 bits
      z[i + 1] += z[i] >> 25;
      z[i] &= LOW_25_BITS;
    }
  };

  // Perform two halves of the carry chain in parallel.
  carry(z, 0);
  carry(z, 4);
  carry(z, 1);
  carry(z, 5);
  carry(z, 2);
  carry(z, 6);
  carry(z, 3);
  carry(z, 7);

  // Since z[3] < 2^64, c < 2^(64-25) = 2^39,
  // so    z[4] < 2^26 + 2^39 < 2^39.0002
  carry(z, 4);
  carry(z, 8);

  // Now z[4] < 2^26
  // and z[5] < 2^25 + 2^13.0002 < 2^25.0004 (good enough)
  // Last carry has a multiplication by 19:
  z[0] += 19 * (z[9] >> 25);
  z[9] &= LOW_25_BITS;

  // Since z[9] < 2^64, c < 2^(64-25) = 2^39,
  //    so z[0] + 19*c < 2^26 + 2^43.248 < 2^43.249.
  carry(z, 0);

  // Now z[1] < 2^25 - 2^(43.249 - 26)
  //          < 2^25.007 (good enough)
  // and we're done.
  h[0] = static_cast<uint32_t>(z[0]);
  h[1] = static_cast<uint32_t>(z[1]);
  h[2] = static_cast<uint32_t>(z[2]);
  h[3] = static_cast<uint32_t>(z[3]);
  h[4] = static_cast<uint32_t>(z[4]);
  h[5] = static_cast<uint32_t>(z[5]);
  h[6] = static_cast<uint32_t>(z[6]);
  h[7] = static_cast<uint32_t>(z[7]);
  h[8] = static_cast<uint32_t>(z[8]);
  h[9] = static_cast<uint32_t>(z[9]);
}
} // namespace sxt::f32b
