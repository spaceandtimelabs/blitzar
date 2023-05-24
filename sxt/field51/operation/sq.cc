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

#include "sxt/field51/operation/sq.h"

#include "sxt/base/type/int.h"
#include "sxt/field51/type/element.h"

namespace sxt::f51o {
//--------------------------------------------------------------------------------------------------
// sq
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void sq(f51t::element& h, const f51t::element& f) noexcept {
  const uint64_t mask = 0x7ffffffffffffULL;
  uint128_t r0, r1, r2, r3, r4;
  uint128_t f0, f1, f2, f3, f4;
  uint128_t f0_2, f1_2, f1_38, f2_38, f3_38, f3_19, f4_19;
  uint64_t r00, r01, r02, r03, r04;
  uint64_t carry;

  f0 = (uint128_t)f[0];
  f1 = (uint128_t)f[1];
  f2 = (uint128_t)f[2];
  f3 = (uint128_t)f[3];
  f4 = (uint128_t)f[4];

  f0_2 = f0 << 1;
  f1_2 = f1 << 1;

  f1_38 = 38ULL * f1;
  f2_38 = 38ULL * f2;
  f3_38 = 38ULL * f3;

  f3_19 = 19ULL * f3;
  f4_19 = 19ULL * f4;

  r0 = f0 * f0 + f1_38 * f4 + f2_38 * f3;
  r1 = f0_2 * f1 + f2_38 * f4 + f3_19 * f3;
  r2 = f0_2 * f2 + f1 * f1 + f3_38 * f4;
  r3 = f0_2 * f3 + f1_2 * f2 + f4_19 * f4;
  r4 = f0_2 * f4 + f1_2 * f3 + f2 * f2;

  r00 = ((uint64_t)r0) & mask;
  carry = (uint64_t)(r0 >> 51);
  r1 += carry;
  r01 = ((uint64_t)r1) & mask;
  carry = (uint64_t)(r1 >> 51);
  r2 += carry;
  r02 = ((uint64_t)r2) & mask;
  carry = (uint64_t)(r2 >> 51);
  r3 += carry;
  r03 = ((uint64_t)r3) & mask;
  carry = (uint64_t)(r3 >> 51);
  r4 += carry;
  r04 = ((uint64_t)r4) & mask;
  carry = (uint64_t)(r4 >> 51);
  r00 += 19ULL * carry;
  carry = r00 >> 51;
  r00 &= mask;
  r01 += carry;
  carry = r01 >> 51;
  r01 &= mask;
  r02 += carry;

  h[0] = r00;
  h[1] = r01;
  h[2] = r02;
  h[3] = r03;
  h[4] = r04;
}

//--------------------------------------------------------------------------------------------------
// sq2
//--------------------------------------------------------------------------------------------------
/*
 h = 2 * f * f
 Can overlap h with f.
*/
CUDA_CALLABLE
void sq2(f51t::element& h, const f51t::element& f) noexcept {
  const uint64_t mask = 0x7ffffffffffffULL;
  uint128_t r0, r1, r2, r3, r4, carry;
  uint64_t f0, f1, f2, f3, f4;
  uint64_t f0_2, f1_2, f1_38, f2_38, f3_38, f3_19, f4_19;
  uint64_t r00, r01, r02, r03, r04;

  f0 = f[0];
  f1 = f[1];
  f2 = f[2];
  f3 = f[3];
  f4 = f[4];

  f0_2 = f0 << 1;
  f1_2 = f1 << 1;

  f1_38 = 38ULL * f1;
  f2_38 = 38ULL * f2;
  f3_38 = 38ULL * f3;

  f3_19 = 19ULL * f3;
  f4_19 = 19ULL * f4;

  r0 = ((uint128_t)f0) * ((uint128_t)f0);
  r0 += ((uint128_t)f1_38) * ((uint128_t)f4);
  r0 += ((uint128_t)f2_38) * ((uint128_t)f3);

  r1 = ((uint128_t)f0_2) * ((uint128_t)f1);
  r1 += ((uint128_t)f2_38) * ((uint128_t)f4);
  r1 += ((uint128_t)f3_19) * ((uint128_t)f3);

  r2 = ((uint128_t)f0_2) * ((uint128_t)f2);
  r2 += ((uint128_t)f1) * ((uint128_t)f1);
  r2 += ((uint128_t)f3_38) * ((uint128_t)f4);

  r3 = ((uint128_t)f0_2) * ((uint128_t)f3);
  r3 += ((uint128_t)f1_2) * ((uint128_t)f2);
  r3 += ((uint128_t)f4_19) * ((uint128_t)f4);

  r4 = ((uint128_t)f0_2) * ((uint128_t)f4);
  r4 += ((uint128_t)f1_2) * ((uint128_t)f3);
  r4 += ((uint128_t)f2) * ((uint128_t)f2);

  r0 <<= 1;
  r1 <<= 1;
  r2 <<= 1;
  r3 <<= 1;
  r4 <<= 1;

  r00 = ((uint64_t)r0) & mask;
  carry = r0 >> 51;
  r1 += carry;
  r01 = ((uint64_t)r1) & mask;
  carry = r1 >> 51;
  r2 += carry;
  r02 = ((uint64_t)r2) & mask;
  carry = r2 >> 51;
  r3 += carry;
  r03 = ((uint64_t)r3) & mask;
  carry = r3 >> 51;
  r4 += carry;
  r04 = ((uint64_t)r4) & mask;
  carry = r4 >> 51;
  r00 += 19ULL * (uint64_t)carry;
  carry = r00 >> 51;
  r00 &= mask;
  r01 += (uint64_t)carry;
  carry = r01 >> 51;
  r01 &= mask;
  r02 += (uint64_t)carry;

  h[0] = r00;
  h[1] = r01;
  h[2] = r02;
  h[3] = r03;
  h[4] = r04;
}
} // namespace sxt::f51o
