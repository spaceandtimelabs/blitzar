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
/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2023
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */
#include "sxt/field51/base/byte_conversion.h"

#include "sxt/base/bit/load.h"
#include "sxt/base/bit/store.h"
#include "sxt/field51/base/reduce.h"

namespace sxt::f51b {
//--------------------------------------------------------------------------------------------------
// from_bytes
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void from_bytes(uint64_t h[5], const uint8_t s[32]) noexcept {
  const uint64_t mask = 0x7ffffffffffffULL;
  uint64_t h0, h1, h2, h3, h4;

  h0 = (basbt::load64_le(s))&mask;
  h1 = (basbt::load64_le(s + 6) >> 3) & mask;
  h2 = (basbt::load64_le(s + 12) >> 6) & mask;
  h3 = (basbt::load64_le(s + 19) >> 1) & mask;
  h4 = (basbt::load64_le(s + 24) >> 12) & mask;

  h[0] = h0;
  h[1] = h1;
  h[2] = h2;
  h[3] = h3;
  h[4] = h4;
}

//--------------------------------------------------------------------------------------------------
// to_bytes
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void to_bytes(uint8_t s[32], const uint64_t h[5]) noexcept {
  uint64_t t[5];
  uint64_t t0, t1, t2, t3;

  reduce(t, h);

  t0 = t[0] | (t[1] << 51);
  t1 = (t[1] >> 13) | (t[2] << 38);
  t2 = (t[2] >> 26) | (t[3] << 25);
  t3 = (t[3] >> 39) | (t[4] << 12);

  basbt::store64_le(s + 0, t0);
  basbt::store64_le(s + 8, t1);
  basbt::store64_le(s + 16, t2);
  basbt::store64_le(s + 24, t3);
}
} // namespace sxt::f51b
