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
#include "sxt/scalar25/operation/mul.h"

#include "sxt/base/bit/load.h"

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// mul
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void mul(s25t::element& s, const s25t::element& a, const s25t::element& b) noexcept {
  // Adopted from libsodium's sc25519_mul
  auto a_data = a.data();
  auto b_data = b.data();
  auto s_data = s.data();

  int64_t a0 = 2097151 & basbt::load_3(a_data);
  int64_t a1 = 2097151 & (basbt::load_4(a_data + 2) >> 5);
  int64_t a2 = 2097151 & (basbt::load_3(a_data + 5) >> 2);
  int64_t a3 = 2097151 & (basbt::load_4(a_data + 7) >> 7);
  int64_t a4 = 2097151 & (basbt::load_4(a_data + 10) >> 4);
  int64_t a5 = 2097151 & (basbt::load_3(a_data + 13) >> 1);
  int64_t a6 = 2097151 & (basbt::load_4(a_data + 15) >> 6);
  int64_t a7 = 2097151 & (basbt::load_3(a_data + 18) >> 3);
  int64_t a8 = 2097151 & basbt::load_3(a_data + 21);
  int64_t a9 = 2097151 & (basbt::load_4(a_data + 23) >> 5);
  int64_t a10 = 2097151 & (basbt::load_3(a_data + 26) >> 2);
  int64_t a11 = (basbt::load_4(a_data + 28) >> 7);

  int64_t b0 = 2097151 & basbt::load_3(b_data);
  int64_t b1 = 2097151 & (basbt::load_4(b_data + 2) >> 5);
  int64_t b2 = 2097151 & (basbt::load_3(b_data + 5) >> 2);
  int64_t b3 = 2097151 & (basbt::load_4(b_data + 7) >> 7);
  int64_t b4 = 2097151 & (basbt::load_4(b_data + 10) >> 4);
  int64_t b5 = 2097151 & (basbt::load_3(b_data + 13) >> 1);
  int64_t b6 = 2097151 & (basbt::load_4(b_data + 15) >> 6);
  int64_t b7 = 2097151 & (basbt::load_3(b_data + 18) >> 3);
  int64_t b8 = 2097151 & basbt::load_3(b_data + 21);
  int64_t b9 = 2097151 & (basbt::load_4(b_data + 23) >> 5);
  int64_t b10 = 2097151 & (basbt::load_3(b_data + 26) >> 2);
  int64_t b11 = (basbt::load_4(b_data + 28) >> 7);

  int64_t s0;
  int64_t s1;
  int64_t s2;
  int64_t s3;
  int64_t s4;
  int64_t s5;
  int64_t s6;
  int64_t s7;
  int64_t s8;
  int64_t s9;
  int64_t s10;
  int64_t s11;
  int64_t s12;
  int64_t s13;
  int64_t s14;
  int64_t s15;
  int64_t s16;
  int64_t s17;
  int64_t s18;
  int64_t s19;
  int64_t s20;
  int64_t s21;
  int64_t s22;
  int64_t s23;

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
  int64_t carry17;
  int64_t carry18;
  int64_t carry19;
  int64_t carry20;
  int64_t carry21;
  int64_t carry22;

  s0 = a0 * b0;
  s1 = a0 * b1 + a1 * b0;
  s2 = a0 * b2 + a1 * b1 + a2 * b0;
  s3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0;
  s4 = a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1 + a4 * b0;
  s5 = a0 * b5 + a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1 + a5 * b0;
  s6 = a0 * b6 + a1 * b5 + a2 * b4 + a3 * b3 + a4 * b2 + a5 * b1 + a6 * b0;
  s7 = a0 * b7 + a1 * b6 + a2 * b5 + a3 * b4 + a4 * b3 + a5 * b2 + a6 * b1 + a7 * b0;
  s8 = a0 * b8 + a1 * b7 + a2 * b6 + a3 * b5 + a4 * b4 + a5 * b3 + a6 * b2 + a7 * b1 + a8 * b0;
  s9 = a0 * b9 + a1 * b8 + a2 * b7 + a3 * b6 + a4 * b5 + a5 * b4 + a6 * b3 + a7 * b2 + a8 * b1 +
       a9 * b0;
  s10 = a0 * b10 + a1 * b9 + a2 * b8 + a3 * b7 + a4 * b6 + a5 * b5 + a6 * b4 + a7 * b3 + a8 * b2 +
        a9 * b1 + a10 * b0;
  s11 = a0 * b11 + a1 * b10 + a2 * b9 + a3 * b8 + a4 * b7 + a5 * b6 + a6 * b5 + a7 * b4 + a8 * b3 +
        a9 * b2 + a10 * b1 + a11 * b0;
  s12 = a1 * b11 + a2 * b10 + a3 * b9 + a4 * b8 + a5 * b7 + a6 * b6 + a7 * b5 + a8 * b4 + a9 * b3 +
        a10 * b2 + a11 * b1;
  s13 = a2 * b11 + a3 * b10 + a4 * b9 + a5 * b8 + a6 * b7 + a7 * b6 + a8 * b5 + a9 * b4 + a10 * b3 +
        a11 * b2;
  s14 = a3 * b11 + a4 * b10 + a5 * b9 + a6 * b8 + a7 * b7 + a8 * b6 + a9 * b5 + a10 * b4 + a11 * b3;
  s15 = a4 * b11 + a5 * b10 + a6 * b9 + a7 * b8 + a8 * b7 + a9 * b6 + a10 * b5 + a11 * b4;
  s16 = a5 * b11 + a6 * b10 + a7 * b9 + a8 * b8 + a9 * b7 + a10 * b6 + a11 * b5;
  s17 = a6 * b11 + a7 * b10 + a8 * b9 + a9 * b8 + a10 * b7 + a11 * b6;
  s18 = a7 * b11 + a8 * b10 + a9 * b9 + a10 * b8 + a11 * b7;
  s19 = a8 * b11 + a9 * b10 + a10 * b9 + a11 * b8;
  s20 = a9 * b11 + a10 * b10 + a11 * b9;
  s21 = a10 * b11 + a11 * b10;
  s22 = a11 * b11;
  s23 = 0;

  carry0 = (s0 + (int64_t)(1L << 20)) >> 21;
  s1 += carry0;
  s0 -= carry0 * ((uint64_t)1L << 21);
  carry2 = (s2 + (int64_t)(1L << 20)) >> 21;
  s3 += carry2;
  s2 -= carry2 * ((uint64_t)1L << 21);
  carry4 = (s4 + (int64_t)(1L << 20)) >> 21;
  s5 += carry4;
  s4 -= carry4 * ((uint64_t)1L << 21);
  carry6 = (s6 + (int64_t)(1L << 20)) >> 21;
  s7 += carry6;
  s6 -= carry6 * ((uint64_t)1L << 21);
  carry8 = (s8 + (int64_t)(1L << 20)) >> 21;
  s9 += carry8;
  s8 -= carry8 * ((uint64_t)1L << 21);
  carry10 = (s10 + (int64_t)(1L << 20)) >> 21;
  s11 += carry10;
  s10 -= carry10 * ((uint64_t)1L << 21);
  carry12 = (s12 + (int64_t)(1L << 20)) >> 21;
  s13 += carry12;
  s12 -= carry12 * ((uint64_t)1L << 21);
  carry14 = (s14 + (int64_t)(1L << 20)) >> 21;
  s15 += carry14;
  s14 -= carry14 * ((uint64_t)1L << 21);
  carry16 = (s16 + (int64_t)(1L << 20)) >> 21;
  s17 += carry16;
  s16 -= carry16 * ((uint64_t)1L << 21);
  carry18 = (s18 + (int64_t)(1L << 20)) >> 21;
  s19 += carry18;
  s18 -= carry18 * ((uint64_t)1L << 21);
  carry20 = (s20 + (int64_t)(1L << 20)) >> 21;
  s21 += carry20;
  s20 -= carry20 * ((uint64_t)1L << 21);
  carry22 = (s22 + (int64_t)(1L << 20)) >> 21;
  s23 += carry22;
  s22 -= carry22 * ((uint64_t)1L << 21);

  carry1 = (s1 + (int64_t)(1L << 20)) >> 21;
  s2 += carry1;
  s1 -= carry1 * ((uint64_t)1L << 21);
  carry3 = (s3 + (int64_t)(1L << 20)) >> 21;
  s4 += carry3;
  s3 -= carry3 * ((uint64_t)1L << 21);
  carry5 = (s5 + (int64_t)(1L << 20)) >> 21;
  s6 += carry5;
  s5 -= carry5 * ((uint64_t)1L << 21);
  carry7 = (s7 + (int64_t)(1L << 20)) >> 21;
  s8 += carry7;
  s7 -= carry7 * ((uint64_t)1L << 21);
  carry9 = (s9 + (int64_t)(1L << 20)) >> 21;
  s10 += carry9;
  s9 -= carry9 * ((uint64_t)1L << 21);
  carry11 = (s11 + (int64_t)(1L << 20)) >> 21;
  s12 += carry11;
  s11 -= carry11 * ((uint64_t)1L << 21);
  carry13 = (s13 + (int64_t)(1L << 20)) >> 21;
  s14 += carry13;
  s13 -= carry13 * ((uint64_t)1L << 21);
  carry15 = (s15 + (int64_t)(1L << 20)) >> 21;
  s16 += carry15;
  s15 -= carry15 * ((uint64_t)1L << 21);
  carry17 = (s17 + (int64_t)(1L << 20)) >> 21;
  s18 += carry17;
  s17 -= carry17 * ((uint64_t)1L << 21);
  carry19 = (s19 + (int64_t)(1L << 20)) >> 21;
  s20 += carry19;
  s19 -= carry19 * ((uint64_t)1L << 21);
  carry21 = (s21 + (int64_t)(1L << 20)) >> 21;
  s22 += carry21;
  s21 -= carry21 * ((uint64_t)1L << 21);

  s11 += s23 * 666643;
  s12 += s23 * 470296;
  s13 += s23 * 654183;
  s14 -= s23 * 997805;
  s15 += s23 * 136657;
  s16 -= s23 * 683901;

  s10 += s22 * 666643;
  s11 += s22 * 470296;
  s12 += s22 * 654183;
  s13 -= s22 * 997805;
  s14 += s22 * 136657;
  s15 -= s22 * 683901;

  s9 += s21 * 666643;
  s10 += s21 * 470296;
  s11 += s21 * 654183;
  s12 -= s21 * 997805;
  s13 += s21 * 136657;
  s14 -= s21 * 683901;

  s8 += s20 * 666643;
  s9 += s20 * 470296;
  s10 += s20 * 654183;
  s11 -= s20 * 997805;
  s12 += s20 * 136657;
  s13 -= s20 * 683901;

  s7 += s19 * 666643;
  s8 += s19 * 470296;
  s9 += s19 * 654183;
  s10 -= s19 * 997805;
  s11 += s19 * 136657;
  s12 -= s19 * 683901;

  s6 += s18 * 666643;
  s7 += s18 * 470296;
  s8 += s18 * 654183;
  s9 -= s18 * 997805;
  s10 += s18 * 136657;
  s11 -= s18 * 683901;

  carry6 = (s6 + (int64_t)(1L << 20)) >> 21;
  s7 += carry6;
  s6 -= carry6 * ((uint64_t)1L << 21);
  carry8 = (s8 + (int64_t)(1L << 20)) >> 21;
  s9 += carry8;
  s8 -= carry8 * ((uint64_t)1L << 21);
  carry10 = (s10 + (int64_t)(1L << 20)) >> 21;
  s11 += carry10;
  s10 -= carry10 * ((uint64_t)1L << 21);
  carry12 = (s12 + (int64_t)(1L << 20)) >> 21;
  s13 += carry12;
  s12 -= carry12 * ((uint64_t)1L << 21);
  carry14 = (s14 + (int64_t)(1L << 20)) >> 21;
  s15 += carry14;
  s14 -= carry14 * ((uint64_t)1L << 21);
  carry16 = (s16 + (int64_t)(1L << 20)) >> 21;
  s17 += carry16;
  s16 -= carry16 * ((uint64_t)1L << 21);

  carry7 = (s7 + (int64_t)(1L << 20)) >> 21;
  s8 += carry7;
  s7 -= carry7 * ((uint64_t)1L << 21);
  carry9 = (s9 + (int64_t)(1L << 20)) >> 21;
  s10 += carry9;
  s9 -= carry9 * ((uint64_t)1L << 21);
  carry11 = (s11 + (int64_t)(1L << 20)) >> 21;
  s12 += carry11;
  s11 -= carry11 * ((uint64_t)1L << 21);
  carry13 = (s13 + (int64_t)(1L << 20)) >> 21;
  s14 += carry13;
  s13 -= carry13 * ((uint64_t)1L << 21);
  carry15 = (s15 + (int64_t)(1L << 20)) >> 21;
  s16 += carry15;
  s15 -= carry15 * ((uint64_t)1L << 21);

  s5 += s17 * 666643;
  s6 += s17 * 470296;
  s7 += s17 * 654183;
  s8 -= s17 * 997805;
  s9 += s17 * 136657;
  s10 -= s17 * 683901;

  s4 += s16 * 666643;
  s5 += s16 * 470296;
  s6 += s16 * 654183;
  s7 -= s16 * 997805;
  s8 += s16 * 136657;
  s9 -= s16 * 683901;

  s3 += s15 * 666643;
  s4 += s15 * 470296;
  s5 += s15 * 654183;
  s6 -= s15 * 997805;
  s7 += s15 * 136657;
  s8 -= s15 * 683901;

  s2 += s14 * 666643;
  s3 += s14 * 470296;
  s4 += s14 * 654183;
  s5 -= s14 * 997805;
  s6 += s14 * 136657;
  s7 -= s14 * 683901;

  s1 += s13 * 666643;
  s2 += s13 * 470296;
  s3 += s13 * 654183;
  s4 -= s13 * 997805;
  s5 += s13 * 136657;
  s6 -= s13 * 683901;

  s0 += s12 * 666643;
  s1 += s12 * 470296;
  s2 += s12 * 654183;
  s3 -= s12 * 997805;
  s4 += s12 * 136657;
  s5 -= s12 * 683901;
  s12 = 0;

  carry0 = (s0 + (int64_t)(1L << 20)) >> 21;
  s1 += carry0;
  s0 -= carry0 * ((uint64_t)1L << 21);
  carry2 = (s2 + (int64_t)(1L << 20)) >> 21;
  s3 += carry2;
  s2 -= carry2 * ((uint64_t)1L << 21);
  carry4 = (s4 + (int64_t)(1L << 20)) >> 21;
  s5 += carry4;
  s4 -= carry4 * ((uint64_t)1L << 21);
  carry6 = (s6 + (int64_t)(1L << 20)) >> 21;
  s7 += carry6;
  s6 -= carry6 * ((uint64_t)1L << 21);
  carry8 = (s8 + (int64_t)(1L << 20)) >> 21;
  s9 += carry8;
  s8 -= carry8 * ((uint64_t)1L << 21);
  carry10 = (s10 + (int64_t)(1L << 20)) >> 21;
  s11 += carry10;
  s10 -= carry10 * ((uint64_t)1L << 21);

  carry1 = (s1 + (int64_t)(1L << 20)) >> 21;
  s2 += carry1;
  s1 -= carry1 * ((uint64_t)1L << 21);
  carry3 = (s3 + (int64_t)(1L << 20)) >> 21;
  s4 += carry3;
  s3 -= carry3 * ((uint64_t)1L << 21);
  carry5 = (s5 + (int64_t)(1L << 20)) >> 21;
  s6 += carry5;
  s5 -= carry5 * ((uint64_t)1L << 21);
  carry7 = (s7 + (int64_t)(1L << 20)) >> 21;
  s8 += carry7;
  s7 -= carry7 * ((uint64_t)1L << 21);
  carry9 = (s9 + (int64_t)(1L << 20)) >> 21;
  s10 += carry9;
  s9 -= carry9 * ((uint64_t)1L << 21);
  carry11 = (s11 + (int64_t)(1L << 20)) >> 21;
  s12 += carry11;
  s11 -= carry11 * ((uint64_t)1L << 21);

  s0 += s12 * 666643;
  s1 += s12 * 470296;
  s2 += s12 * 654183;
  s3 -= s12 * 997805;
  s4 += s12 * 136657;
  s5 -= s12 * 683901;
  s12 = 0;

  carry0 = s0 >> 21;
  s1 += carry0;
  s0 -= carry0 * ((uint64_t)1L << 21);
  carry1 = s1 >> 21;
  s2 += carry1;
  s1 -= carry1 * ((uint64_t)1L << 21);
  carry2 = s2 >> 21;
  s3 += carry2;
  s2 -= carry2 * ((uint64_t)1L << 21);
  carry3 = s3 >> 21;
  s4 += carry3;
  s3 -= carry3 * ((uint64_t)1L << 21);
  carry4 = s4 >> 21;
  s5 += carry4;
  s4 -= carry4 * ((uint64_t)1L << 21);
  carry5 = s5 >> 21;
  s6 += carry5;
  s5 -= carry5 * ((uint64_t)1L << 21);
  carry6 = s6 >> 21;
  s7 += carry6;
  s6 -= carry6 * ((uint64_t)1L << 21);
  carry7 = s7 >> 21;
  s8 += carry7;
  s7 -= carry7 * ((uint64_t)1L << 21);
  carry8 = s8 >> 21;
  s9 += carry8;
  s8 -= carry8 * ((uint64_t)1L << 21);
  carry9 = s9 >> 21;
  s10 += carry9;
  s9 -= carry9 * ((uint64_t)1L << 21);
  carry10 = s10 >> 21;
  s11 += carry10;
  s10 -= carry10 * ((uint64_t)1L << 21);
  carry11 = s11 >> 21;
  s12 += carry11;
  s11 -= carry11 * ((uint64_t)1L << 21);

  s0 += s12 * 666643;
  s1 += s12 * 470296;
  s2 += s12 * 654183;
  s3 -= s12 * 997805;
  s4 += s12 * 136657;
  s5 -= s12 * 683901;

  carry0 = s0 >> 21;
  s1 += carry0;
  s0 -= carry0 * ((uint64_t)1L << 21);
  carry1 = s1 >> 21;
  s2 += carry1;
  s1 -= carry1 * ((uint64_t)1L << 21);
  carry2 = s2 >> 21;
  s3 += carry2;
  s2 -= carry2 * ((uint64_t)1L << 21);
  carry3 = s3 >> 21;
  s4 += carry3;
  s3 -= carry3 * ((uint64_t)1L << 21);
  carry4 = s4 >> 21;
  s5 += carry4;
  s4 -= carry4 * ((uint64_t)1L << 21);
  carry5 = s5 >> 21;
  s6 += carry5;
  s5 -= carry5 * ((uint64_t)1L << 21);
  carry6 = s6 >> 21;
  s7 += carry6;
  s6 -= carry6 * ((uint64_t)1L << 21);
  carry7 = s7 >> 21;
  s8 += carry7;
  s7 -= carry7 * ((uint64_t)1L << 21);
  carry8 = s8 >> 21;
  s9 += carry8;
  s8 -= carry8 * ((uint64_t)1L << 21);
  carry9 = s9 >> 21;
  s10 += carry9;
  s9 -= carry9 * ((uint64_t)1L << 21);
  carry10 = s10 >> 21;
  s11 += carry10;
  s10 -= carry10 * ((uint64_t)1L << 21);

  s_data[0] = s0 >> 0;
  s_data[1] = s0 >> 8;
  s_data[2] = (s0 >> 16) | (s1 * ((uint64_t)1 << 5));
  s_data[3] = s1 >> 3;
  s_data[4] = s1 >> 11;
  s_data[5] = (s1 >> 19) | (s2 * ((uint64_t)1 << 2));
  s_data[6] = s2 >> 6;
  s_data[7] = (s2 >> 14) | (s3 * ((uint64_t)1 << 7));
  s_data[8] = s3 >> 1;
  s_data[9] = s3 >> 9;
  s_data[10] = (s3 >> 17) | (s4 * ((uint64_t)1 << 4));
  s_data[11] = s4 >> 4;
  s_data[12] = s4 >> 12;
  s_data[13] = (s4 >> 20) | (s5 * ((uint64_t)1 << 1));
  s_data[14] = s5 >> 7;
  s_data[15] = (s5 >> 15) | (s6 * ((uint64_t)1 << 6));
  s_data[16] = s6 >> 2;
  s_data[17] = s6 >> 10;
  s_data[18] = (s6 >> 18) | (s7 * ((uint64_t)1 << 3));
  s_data[19] = s7 >> 5;
  s_data[20] = s7 >> 13;
  s_data[21] = s8 >> 0;
  s_data[22] = s8 >> 8;
  s_data[23] = (s8 >> 16) | (s9 * ((uint64_t)1 << 5));
  s_data[24] = s9 >> 3;
  s_data[25] = s9 >> 11;
  s_data[26] = (s9 >> 19) | (s10 * ((uint64_t)1 << 2));
  s_data[27] = s10 >> 6;
  s_data[28] = (s10 >> 14) | (s11 * ((uint64_t)1 << 7));
  s_data[29] = s11 >> 1;
  s_data[30] = s11 >> 9;
  s_data[31] = s11 >> 17;
}
} // namespace sxt::s25o
