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
#include "sxt/scalar25/base/reduce.h"

#include "sxt/base/bit/load.h"

namespace sxt::s25b {
//--------------------------------------------------------------------------------------------------
// reduce_impl
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void reduce_impl(uint8_t dest[32], int64_t s[24]) noexcept {
  int64_t carry0;
  int64_t carry1;
  int64_t carry2;
  int64_t carry3;
  int64_t carry4;
  int64_t carry5;
  int64_t carry6;
  int64_t carry7;
  int64_t carry8;
  int64_t carry9;
  int64_t carry10;
  int64_t carry11;
  int64_t carry12;
  int64_t carry13;
  int64_t carry14;
  int64_t carry15;
  int64_t carry16;

  s[11] += s[23] * 666643;
  s[12] += s[23] * 470296;
  s[13] += s[23] * 654183;
  s[14] -= s[23] * 997805;
  s[15] += s[23] * 136657;
  s[16] -= s[23] * 683901;

  s[10] += s[22] * 666643;
  s[11] += s[22] * 470296;
  s[12] += s[22] * 654183;
  s[13] -= s[22] * 997805;
  s[14] += s[22] * 136657;
  s[15] -= s[22] * 683901;

  s[9] += s[21] * 666643;
  s[10] += s[21] * 470296;
  s[11] += s[21] * 654183;
  s[12] -= s[21] * 997805;
  s[13] += s[21] * 136657;
  s[14] -= s[21] * 683901;

  s[8] += s[20] * 666643;
  s[9] += s[20] * 470296;
  s[10] += s[20] * 654183;
  s[11] -= s[20] * 997805;
  s[12] += s[20] * 136657;
  s[13] -= s[20] * 683901;

  s[7] += s[19] * 666643;
  s[8] += s[19] * 470296;
  s[9] += s[19] * 654183;
  s[10] -= s[19] * 997805;
  s[11] += s[19] * 136657;
  s[12] -= s[19] * 683901;

  s[6] += s[18] * 666643;
  s[7] += s[18] * 470296;
  s[8] += s[18] * 654183;
  s[9] -= s[18] * 997805;
  s[10] += s[18] * 136657;
  s[11] -= s[18] * 683901;

  carry6 = (s[6] + (int64_t)(1L << 20)) >> 21;
  s[7] += carry6;
  s[6] -= carry6 * ((uint64_t)1L << 21);
  carry8 = (s[8] + (int64_t)(1L << 20)) >> 21;
  s[9] += carry8;
  s[8] -= carry8 * ((uint64_t)1L << 21);
  carry10 = (s[10] + (int64_t)(1L << 20)) >> 21;
  s[11] += carry10;
  s[10] -= carry10 * ((uint64_t)1L << 21);
  carry12 = (s[12] + (int64_t)(1L << 20)) >> 21;
  s[13] += carry12;
  s[12] -= carry12 * ((uint64_t)1L << 21);
  carry14 = (s[14] + (int64_t)(1L << 20)) >> 21;
  s[15] += carry14;
  s[14] -= carry14 * ((uint64_t)1L << 21);
  carry16 = (s[16] + (int64_t)(1L << 20)) >> 21;
  s[17] += carry16;
  s[16] -= carry16 * ((uint64_t)1L << 21);

  carry7 = (s[7] + (int64_t)(1L << 20)) >> 21;
  s[8] += carry7;
  s[7] -= carry7 * ((uint64_t)1L << 21);
  carry9 = (s[9] + (int64_t)(1L << 20)) >> 21;
  s[10] += carry9;
  s[9] -= carry9 * ((uint64_t)1L << 21);
  carry11 = (s[11] + (int64_t)(1L << 20)) >> 21;
  s[12] += carry11;
  s[11] -= carry11 * ((uint64_t)1L << 21);
  carry13 = (s[13] + (int64_t)(1L << 20)) >> 21;
  s[14] += carry13;
  s[13] -= carry13 * ((uint64_t)1L << 21);
  carry15 = (s[15] + (int64_t)(1L << 20)) >> 21;
  s[16] += carry15;
  s[15] -= carry15 * ((uint64_t)1L << 21);

  s[5] += s[17] * 666643;
  s[6] += s[17] * 470296;
  s[7] += s[17] * 654183;
  s[8] -= s[17] * 997805;
  s[9] += s[17] * 136657;
  s[10] -= s[17] * 683901;

  s[4] += s[16] * 666643;
  s[5] += s[16] * 470296;
  s[6] += s[16] * 654183;
  s[7] -= s[16] * 997805;
  s[8] += s[16] * 136657;
  s[9] -= s[16] * 683901;

  s[3] += s[15] * 666643;
  s[4] += s[15] * 470296;
  s[5] += s[15] * 654183;
  s[6] -= s[15] * 997805;
  s[7] += s[15] * 136657;
  s[8] -= s[15] * 683901;

  s[2] += s[14] * 666643;
  s[3] += s[14] * 470296;
  s[4] += s[14] * 654183;
  s[5] -= s[14] * 997805;
  s[6] += s[14] * 136657;
  s[7] -= s[14] * 683901;

  s[1] += s[13] * 666643;
  s[2] += s[13] * 470296;
  s[3] += s[13] * 654183;
  s[4] -= s[13] * 997805;
  s[5] += s[13] * 136657;
  s[6] -= s[13] * 683901;

  s[0] += s[12] * 666643;
  s[1] += s[12] * 470296;
  s[2] += s[12] * 654183;
  s[3] -= s[12] * 997805;
  s[4] += s[12] * 136657;
  s[5] -= s[12] * 683901;
  s[12] = 0;

  carry0 = (s[0] + (int64_t)(1L << 20)) >> 21;
  s[1] += carry0;
  s[0] -= carry0 * ((uint64_t)1L << 21);
  carry2 = (s[2] + (int64_t)(1L << 20)) >> 21;
  s[3] += carry2;
  s[2] -= carry2 * ((uint64_t)1L << 21);
  carry4 = (s[4] + (int64_t)(1L << 20)) >> 21;
  s[5] += carry4;
  s[4] -= carry4 * ((uint64_t)1L << 21);
  carry6 = (s[6] + (int64_t)(1L << 20)) >> 21;
  s[7] += carry6;
  s[6] -= carry6 * ((uint64_t)1L << 21);
  carry8 = (s[8] + (int64_t)(1L << 20)) >> 21;
  s[9] += carry8;
  s[8] -= carry8 * ((uint64_t)1L << 21);
  carry10 = (s[10] + (int64_t)(1L << 20)) >> 21;
  s[11] += carry10;
  s[10] -= carry10 * ((uint64_t)1L << 21);

  carry1 = (s[1] + (int64_t)(1L << 20)) >> 21;
  s[2] += carry1;
  s[1] -= carry1 * ((uint64_t)1L << 21);
  carry3 = (s[3] + (int64_t)(1L << 20)) >> 21;
  s[4] += carry3;
  s[3] -= carry3 * ((uint64_t)1L << 21);
  carry5 = (s[5] + (int64_t)(1L << 20)) >> 21;
  s[6] += carry5;
  s[5] -= carry5 * ((uint64_t)1L << 21);
  carry7 = (s[7] + (int64_t)(1L << 20)) >> 21;
  s[8] += carry7;
  s[7] -= carry7 * ((uint64_t)1L << 21);
  carry9 = (s[9] + (int64_t)(1L << 20)) >> 21;
  s[10] += carry9;
  s[9] -= carry9 * ((uint64_t)1L << 21);
  carry11 = (s[11] + (int64_t)(1L << 20)) >> 21;
  s[12] += carry11;
  s[11] -= carry11 * ((uint64_t)1L << 21);

  s[0] += s[12] * 666643;
  s[1] += s[12] * 470296;
  s[2] += s[12] * 654183;
  s[3] -= s[12] * 997805;
  s[4] += s[12] * 136657;
  s[5] -= s[12] * 683901;
  s[12] = 0;

  carry0 = s[0] >> 21;
  s[1] += carry0;
  s[0] -= carry0 * ((uint64_t)1L << 21);
  carry1 = s[1] >> 21;
  s[2] += carry1;
  s[1] -= carry1 * ((uint64_t)1L << 21);
  carry2 = s[2] >> 21;
  s[3] += carry2;
  s[2] -= carry2 * ((uint64_t)1L << 21);
  carry3 = s[3] >> 21;
  s[4] += carry3;
  s[3] -= carry3 * ((uint64_t)1L << 21);
  carry4 = s[4] >> 21;
  s[5] += carry4;
  s[4] -= carry4 * ((uint64_t)1L << 21);
  carry5 = s[5] >> 21;
  s[6] += carry5;
  s[5] -= carry5 * ((uint64_t)1L << 21);
  carry6 = s[6] >> 21;
  s[7] += carry6;
  s[6] -= carry6 * ((uint64_t)1L << 21);
  carry7 = s[7] >> 21;
  s[8] += carry7;
  s[7] -= carry7 * ((uint64_t)1L << 21);
  carry8 = s[8] >> 21;
  s[9] += carry8;
  s[8] -= carry8 * ((uint64_t)1L << 21);
  carry9 = s[9] >> 21;
  s[10] += carry9;
  s[9] -= carry9 * ((uint64_t)1L << 21);
  carry10 = s[10] >> 21;
  s[11] += carry10;
  s[10] -= carry10 * ((uint64_t)1L << 21);
  carry11 = s[11] >> 21;
  s[12] += carry11;
  s[11] -= carry11 * ((uint64_t)1L << 21);

  s[0] += s[12] * 666643;
  s[1] += s[12] * 470296;
  s[2] += s[12] * 654183;
  s[3] -= s[12] * 997805;
  s[4] += s[12] * 136657;
  s[5] -= s[12] * 683901;

  carry0 = s[0] >> 21;
  s[1] += carry0;
  s[0] -= carry0 * ((uint64_t)1L << 21);
  carry1 = s[1] >> 21;
  s[2] += carry1;
  s[1] -= carry1 * ((uint64_t)1L << 21);
  carry2 = s[2] >> 21;
  s[3] += carry2;
  s[2] -= carry2 * ((uint64_t)1L << 21);
  carry3 = s[3] >> 21;
  s[4] += carry3;
  s[3] -= carry3 * ((uint64_t)1L << 21);
  carry4 = s[4] >> 21;
  s[5] += carry4;
  s[4] -= carry4 * ((uint64_t)1L << 21);
  carry5 = s[5] >> 21;
  s[6] += carry5;
  s[5] -= carry5 * ((uint64_t)1L << 21);
  carry6 = s[6] >> 21;
  s[7] += carry6;
  s[6] -= carry6 * ((uint64_t)1L << 21);
  carry7 = s[7] >> 21;
  s[8] += carry7;
  s[7] -= carry7 * ((uint64_t)1L << 21);
  carry8 = s[8] >> 21;
  s[9] += carry8;
  s[8] -= carry8 * ((uint64_t)1L << 21);
  carry9 = s[9] >> 21;
  s[10] += carry9;
  s[9] -= carry9 * ((uint64_t)1L << 21);
  carry10 = s[10] >> 21;
  s[11] += carry10;
  s[10] -= carry10 * ((uint64_t)1L << 21);

  dest[0] = s[0] >> 0;
  dest[1] = s[0] >> 8;
  dest[2] = (s[0] >> 16) | (s[1] * ((uint64_t)1 << 5));
  dest[3] = s[1] >> 3;
  dest[4] = s[1] >> 11;
  dest[5] = (s[1] >> 19) | (s[2] * ((uint64_t)1 << 2));
  dest[6] = s[2] >> 6;
  dest[7] = (s[2] >> 14) | (s[3] * ((uint64_t)1 << 7));
  dest[8] = s[3] >> 1;
  dest[9] = s[3] >> 9;
  dest[10] = (s[3] >> 17) | (s[4] * ((uint64_t)1 << 4));
  dest[11] = s[4] >> 4;
  dest[12] = s[4] >> 12;
  dest[13] = (s[4] >> 20) | (s[5] * ((uint64_t)1 << 1));
  dest[14] = s[5] >> 7;
  dest[15] = (s[5] >> 15) | (s[6] * ((uint64_t)1 << 6));
  dest[16] = s[6] >> 2;
  dest[17] = s[6] >> 10;
  dest[18] = (s[6] >> 18) | (s[7] * ((uint64_t)1 << 3));
  dest[19] = s[7] >> 5;
  dest[20] = s[7] >> 13;
  dest[21] = s[8] >> 0;
  dest[22] = s[8] >> 8;
  dest[23] = (s[8] >> 16) | (s[9] * ((uint64_t)1 << 5));
  dest[24] = s[9] >> 3;
  dest[25] = s[9] >> 11;
  dest[26] = (s[9] >> 19) | (s[10] * ((uint64_t)1 << 2));
  dest[27] = s[10] >> 6;
  dest[28] = (s[10] >> 14) | (s[11] * ((uint64_t)1 << 7));
  dest[29] = s[11] >> 1;
  dest[30] = s[11] >> 9;
  dest[31] = s[11] >> 17;
}

//--------------------------------------------------------------------------------------------------
// reduce33
//--------------------------------------------------------------------------------------------------
//
// Modified from libsodium's sc25519_reduce which reduces a 64-byte array by
// reading s[32] to s[63] out as zero.
CUDA_CALLABLE
void reduce33(uint8_t dest[32], uint8_t byte32) noexcept {
  uint64_t last_two_bytes = static_cast<uint64_t>(dest[31]) | (static_cast<uint64_t>(byte32) << 8);

  int64_t reduce_data[24] = {
      2097151LL & static_cast<int64_t>(basbt::load_3(dest)),
      2097151LL & static_cast<int64_t>(basbt::load_4(dest + 2) >> 5),
      2097151LL & static_cast<int64_t>(basbt::load_3(dest + 5) >> 2),
      2097151LL & static_cast<int64_t>(basbt::load_4(dest + 7) >> 7),
      2097151LL & static_cast<int64_t>(basbt::load_4(dest + 10) >> 4),
      2097151LL & static_cast<int64_t>(basbt::load_3(dest + 13) >> 1),
      2097151LL & static_cast<int64_t>(basbt::load_4(dest + 15) >> 6),
      2097151LL & static_cast<int64_t>(basbt::load_3(dest + 18) >> 3),
      2097151LL & static_cast<int64_t>(basbt::load_3(dest + 21)),
      2097151LL & static_cast<int64_t>(basbt::load_4(dest + 23) >> 5),
      2097151LL & static_cast<int64_t>(basbt::load_3(dest + 26) >> 2),
      2097151LL & static_cast<int64_t>(basbt::load_4(dest + 28) >> 7),
      2097151LL & static_cast<int64_t>(last_two_bytes >> 4),
      0LL,
  };

  reduce_impl(dest, reduce_data);
}
} // namespace sxt::s25b
