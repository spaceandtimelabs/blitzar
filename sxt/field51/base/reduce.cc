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
#include "sxt/field51/base/reduce.h"

namespace sxt::f51b {
//--------------------------------------------------------------------------------------------------
// reduce
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void reduce(uint64_t h[5], const uint64_t f[5]) noexcept {
  const uint64_t mask = 0x7ffffffffffffULL;
  uint128_t t[5];

  t[0] = f[0];
  t[1] = f[1];
  t[2] = f[2];
  t[3] = f[3];
  t[4] = f[4];

  t[1] += t[0] >> 51;
  t[0] &= mask;
  t[2] += t[1] >> 51;
  t[1] &= mask;
  t[3] += t[2] >> 51;
  t[2] &= mask;
  t[4] += t[3] >> 51;
  t[3] &= mask;
  t[0] += 19 * (t[4] >> 51);
  t[4] &= mask;

  t[1] += t[0] >> 51;
  t[0] &= mask;
  t[2] += t[1] >> 51;
  t[1] &= mask;
  t[3] += t[2] >> 51;
  t[2] &= mask;
  t[4] += t[3] >> 51;
  t[3] &= mask;
  t[0] += 19 * (t[4] >> 51);
  t[4] &= mask;

  /* now t is between 0 and 2^255-1, properly carried. */
  /* case 1: between 0 and 2^255-20. case 2: between 2^255-19 and 2^255-1. */

  t[0] += 19ULL;

  t[1] += t[0] >> 51;
  t[0] &= mask;
  t[2] += t[1] >> 51;
  t[1] &= mask;
  t[3] += t[2] >> 51;
  t[2] &= mask;
  t[4] += t[3] >> 51;
  t[3] &= mask;
  t[0] += 19ULL * (t[4] >> 51);
  t[4] &= mask;

  /* now between 19 and 2^255-1 in both cases, and offset by 19. */

  t[0] += 0x8000000000000 - 19ULL;
  t[1] += 0x8000000000000 - 1ULL;
  t[2] += 0x8000000000000 - 1ULL;
  t[3] += 0x8000000000000 - 1ULL;
  t[4] += 0x8000000000000 - 1ULL;

  /* now between 2^255 and 2^256-20, and offset by 2^255. */

  t[1] += t[0] >> 51;
  t[0] &= mask;
  t[2] += t[1] >> 51;
  t[1] &= mask;
  t[3] += t[2] >> 51;
  t[2] &= mask;
  t[4] += t[3] >> 51;
  t[3] &= mask;
  t[4] &= mask;

  h[0] = t[0];
  h[1] = t[1];
  h[2] = t[2];
  h[3] = t[3];
  h[4] = t[4];
}
} // namespace sxt::f51b
