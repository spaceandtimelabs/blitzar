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
#include "sxt/field32/base/byte_conversion.h"

#include "sxt/base/bit/load.h"
#include "sxt/base/bit/store.h"
#include "sxt/field32/base/reduce.h"

namespace sxt::f32b {
//--------------------------------------------------------------------------------------------------
// from_bytes
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void from_bytes(uint32_t h[10], const uint8_t s[32]) noexcept {
  auto load3 = [](const uint8_t b[3]) -> uint64_t {
    return (static_cast<uint64_t>(b[0]) | (static_cast<uint64_t>(b[1]) << 8) |
            (static_cast<uint64_t>(b[2]) << 16));
  };

  auto load4 = [](const uint8_t b[4]) -> uint64_t {
    return (static_cast<uint64_t>(b[0]) | (static_cast<uint64_t>(b[1]) << 8) |
            (static_cast<uint64_t>(b[2]) << 16) | (static_cast<uint64_t>(b[3]) << 24));
  };

  uint64_t t[10];
  constexpr uint64_t LOW_23_BITS = (1ULL << 23) - 1;

  t[0] = load4(s);
  t[1] = load3(s + 4) << 6;
  t[2] = load3(s + 7) << 5;
  t[3] = load3(s + 10) << 3;
  t[4] = load3(s + 13) << 2;
  t[5] = load4(s + 16);
  t[6] = load3(s + 20) << 7;
  t[7] = load3(s + 23) << 5;
  t[8] = load3(s + 26) << 4;
  t[9] = (load3(s + 29) & LOW_23_BITS) << 2;

  reduce(h, t);
}

//--------------------------------------------------------------------------------------------------
// to_bytes
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void to_bytes(uint8_t s[32], const uint32_t h[10]) noexcept {
  // Reduce the value represented by `in` to the range [0,2*p)
  uint64_t inp[10];
  for (size_t i = 0; i < 10; ++i) {
    inp[i] = static_cast<uint32_t>(h[i]);
  }

  uint32_t h_tmp[10];
  reduce(h_tmp, inp);

  // Let h be the value to encode.
  //
  // Write h = pq + r with 0 <= r < p.  We want to compute r = h mod p.
  //
  // Since h < 2*p, q = 0 or 1, with q = 0 when h < p and q = 1 when h >= p.
  //
  // Notice that h >= p <==> h + 19 >= p + 19 <==> h + 19 >= 2^255.
  // Therefore q can be computed as the carry bit of h + 19.
  uint32_t q = (h_tmp[0] + 19) >> 26;
  q = (h_tmp[1] + q) >> 25;
  q = (h_tmp[2] + q) >> 26;
  q = (h_tmp[3] + q) >> 25;
  q = (h_tmp[4] + q) >> 26;
  q = (h_tmp[5] + q) >> 25;
  q = (h_tmp[6] + q) >> 26;
  q = (h_tmp[7] + q) >> 25;
  q = (h_tmp[8] + q) >> 26;
  q = (h_tmp[9] + q) >> 25;

  // debug_assert!(q == 0 || q == 1);

  // Now we can compute r as r = h - pq = r - (2^255-19)q = r + 19q - 2^255q

  constexpr uint32_t LOW_25_BITS = (1 << 25) - 1;
  constexpr uint32_t LOW_26_BITS = (1 << 26) - 1;

  h_tmp[0] += 19 * q;

  // Now carry the result to compute r + 19q...
  h_tmp[1] += h_tmp[0] >> 26;
  h_tmp[0] &= LOW_26_BITS;
  h_tmp[2] += h_tmp[1] >> 25;
  h_tmp[1] &= LOW_25_BITS;
  h_tmp[3] += h_tmp[2] >> 26;
  h_tmp[2] &= LOW_26_BITS;
  h_tmp[4] += h_tmp[3] >> 25;
  h_tmp[3] &= LOW_25_BITS;
  h_tmp[5] += h_tmp[4] >> 26;
  h_tmp[4] &= LOW_26_BITS;
  h_tmp[6] += h_tmp[5] >> 25;
  h_tmp[5] &= LOW_25_BITS;
  h_tmp[7] += h_tmp[6] >> 26;
  h_tmp[6] &= LOW_26_BITS;
  h_tmp[8] += h_tmp[7] >> 25;
  h_tmp[7] &= LOW_25_BITS;
  h_tmp[9] += h_tmp[8] >> 26;
  h_tmp[8] &= LOW_26_BITS;

  // ... but instead of carrying the value
  // (h[9] >> 25) = q*2^255 into another limb,
  // discard it, subtracting the value from h.
  // debug_assert!((h[9] >> 25) == 0 || (h[9] >> 25) == 1);
  h_tmp[9] &= LOW_25_BITS;

  s[0] = static_cast<uint8_t>(h_tmp[0] >> 0);
  s[1] = static_cast<uint8_t>(h_tmp[0] >> 8);
  s[2] = static_cast<uint8_t>(h_tmp[0] >> 16);
  s[3] = static_cast<uint8_t>((h_tmp[0] >> 24) | (h_tmp[1] << 2));
  s[4] = static_cast<uint8_t>(h_tmp[1] >> 6);
  s[5] = static_cast<uint8_t>(h_tmp[1] >> 14);
  s[6] = static_cast<uint8_t>((h_tmp[1] >> 22) | (h_tmp[2] << 3));
  s[7] = static_cast<uint8_t>(h_tmp[2] >> 5);
  s[8] = static_cast<uint8_t>(h_tmp[2] >> 13);
  s[9] = static_cast<uint8_t>((h_tmp[2] >> 21) | (h_tmp[3] << 5));
  s[10] = static_cast<uint8_t>(h_tmp[3] >> 3);
  s[11] = static_cast<uint8_t>(h_tmp[3] >> 11);
  s[12] = static_cast<uint8_t>((h_tmp[3] >> 19) | (h_tmp[4] << 6));
  s[13] = static_cast<uint8_t>(h_tmp[4] >> 2);
  s[14] = static_cast<uint8_t>(h_tmp[4] >> 10);
  s[15] = static_cast<uint8_t>(h_tmp[4] >> 18);
  s[16] = static_cast<uint8_t>(h_tmp[5] >> 0);
  s[17] = static_cast<uint8_t>(h_tmp[5] >> 8);
  s[18] = static_cast<uint8_t>(h_tmp[5] >> 16);
  s[19] = static_cast<uint8_t>((h_tmp[5] >> 24) | (h_tmp[6] << 1));
  s[20] = static_cast<uint8_t>(h_tmp[6] >> 7);
  s[21] = static_cast<uint8_t>(h_tmp[6] >> 15);
  s[22] = static_cast<uint8_t>((h_tmp[6] >> 23) | (h_tmp[7] << 3));
  s[23] = static_cast<uint8_t>(h_tmp[7] >> 5);
  s[24] = static_cast<uint8_t>(h_tmp[7] >> 13);
  s[25] = static_cast<uint8_t>((h_tmp[7] >> 21) | (h_tmp[8] << 4));
  s[26] = static_cast<uint8_t>(h_tmp[8] >> 4);
  s[27] = static_cast<uint8_t>(h_tmp[8] >> 12);
  s[28] = static_cast<uint8_t>((h_tmp[8] >> 20) | (h_tmp[9] << 6));
  s[29] = static_cast<uint8_t>(h_tmp[9] >> 2);
  s[30] = static_cast<uint8_t>(h_tmp[9] >> 10);
  s[31] = static_cast<uint8_t>(h_tmp[9] >> 18);

  // Check that high bit is cleared
  // debug_assert!((s[31] & 0b1000_0000u8) == 0u8);
}
} // namespace sxt::f32b
