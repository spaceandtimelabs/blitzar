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

#include "sxt/field32/operation/mul.h"

#include "sxt/base/type/int.h"
#include "sxt/field32/base/reduce.h"
#include "sxt/field32/type/element.h"

namespace sxt::f32o {
//--------------------------------------------------------------------------------------------------
// mul
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void mul(f32t::element& h, const f32t::element& f, const f32t::element& g) noexcept {
  // Helper function to multiply two 32-bit integers with 64 bits of output.
  auto m = [](uint32_t x, uint32_t y) -> uint64_t {
    return static_cast<uint64_t>(x) * static_cast<uint64_t>(y);
  };

  // let x: &[u32; 10] = &self.0;
  uint32_t x[10] = {f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9]};
  // let y: &[u32; 10] = &_rhs.0;
  uint32_t y[10] = {g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], g[9]};

  // We assume that the input limbs x[i], y[i] are bounded by:
  //
  // x[i], y[i] < 2^(26 + b) if i even
  // x[i], y[i] < 2^(25 + b) if i odd
  //
  // where b is a (real) parameter representing the excess bits of
  // the limbs.  We track the bitsizes of all variables through
  // the computation and solve at the end for the allowable
  // headroom bitsize b (which determines how many additions we
  // can perform between reductions or multiplications).

  uint32_t y1_19 = 19 * y[1]; // This fits in a u32
  uint32_t y2_19 = 19 * y[2]; // iff 26 + b + lg(19) < 32
  uint32_t y3_19 = 19 * y[3]; // if  b < 32 - 26 - 4.248 = 1.752
  uint32_t y4_19 = 19 * y[4];
  uint32_t y5_19 = 19 * y[5]; // below, b<2.5: this is a bottleneck,
  uint32_t y6_19 = 19 * y[6]; // could be avoided by promoting to
  uint32_t y7_19 = 19 * y[7]; // u64 here instead of in m()
  uint32_t y8_19 = 19 * y[8];
  uint32_t y9_19 = 19 * y[9];

  // What happens when we multiply x[i] with y[j] and place the
  // result into the (i+j)-th limb?
  //
  // x[i]      represents the value x[i]*2^ceil(i*51/2)
  // y[j]      represents the value y[j]*2^ceil(j*51/2)
  // z[i+j]    represents the value z[i+j]*2^ceil((i+j)*51/2)
  // x[i]*y[j] represents the value x[i]*y[i]*2^(ceil(i*51/2)+ceil(j*51/2))
  //
  // Since the radix is already accounted for, the result placed
  // into the (i+j)-th limb should be
  //
  // x[i]*y[i]*2^(ceil(i*51/2)+ceil(j*51/2) - ceil((i+j)*51/2)).
  //
  // The value of ceil(i*51/2)+ceil(j*51/2) - ceil((i+j)*51/2) is
  // 1 when both i and j are odd, and 0 otherwise.  So we add
  //
  //   x[i]*y[j] if either i or j is even
  // 2*x[i]*y[j] if i and j are both odd
  //
  // by using precomputed multiples of x[i] for odd i:

  uint32_t x1_2 = 2 * x[1]; // This fits in a u32 iff 25 + b + 1 < 32
  uint32_t x3_2 = 2 * x[3]; //                    iff b < 6
  uint32_t x5_2 = 2 * x[5];
  uint32_t x7_2 = 2 * x[7];
  uint32_t x9_2 = 2 * x[9];

  uint64_t z[10];
  z[0] = m(x[0], y[0]) + m(x1_2, y9_19) + m(x[2], y8_19) + m(x3_2, y7_19) + m(x[4], y6_19) +
         m(x5_2, y5_19) + m(x[6], y4_19) + m(x7_2, y3_19) + m(x[8], y2_19) + m(x9_2, y1_19);
  z[1] = m(x[0], y[1]) + m(x[1], y[0]) + m(x[2], y9_19) + m(x[3], y8_19) + m(x[4], y7_19) +
         m(x[5], y6_19) + m(x[6], y5_19) + m(x[7], y4_19) + m(x[8], y3_19) + m(x[9], y2_19);
  z[2] = m(x[0], y[2]) + m(x1_2, y[1]) + m(x[2], y[0]) + m(x3_2, y9_19) + m(x[4], y8_19) +
         m(x5_2, y7_19) + m(x[6], y6_19) + m(x7_2, y5_19) + m(x[8], y4_19) + m(x9_2, y3_19);
  z[3] = m(x[0], y[3]) + m(x[1], y[2]) + m(x[2], y[1]) + m(x[3], y[0]) + m(x[4], y9_19) +
         m(x[5], y8_19) + m(x[6], y7_19) + m(x[7], y6_19) + m(x[8], y5_19) + m(x[9], y4_19);
  z[4] = m(x[0], y[4]) + m(x1_2, y[3]) + m(x[2], y[2]) + m(x3_2, y[1]) + m(x[4], y[0]) +
         m(x5_2, y9_19) + m(x[6], y8_19) + m(x7_2, y7_19) + m(x[8], y6_19) + m(x9_2, y5_19);
  z[5] = m(x[0], y[5]) + m(x[1], y[4]) + m(x[2], y[3]) + m(x[3], y[2]) + m(x[4], y[1]) +
         m(x[5], y[0]) + m(x[6], y9_19) + m(x[7], y8_19) + m(x[8], y7_19) + m(x[9], y6_19);
  z[6] = m(x[0], y[6]) + m(x1_2, y[5]) + m(x[2], y[4]) + m(x3_2, y[3]) + m(x[4], y[2]) +
         m(x5_2, y[1]) + m(x[6], y[0]) + m(x7_2, y9_19) + m(x[8], y8_19) + m(x9_2, y7_19);
  z[7] = m(x[0], y[7]) + m(x[1], y[6]) + m(x[2], y[5]) + m(x[3], y[4]) + m(x[4], y[3]) +
         m(x[5], y[2]) + m(x[6], y[1]) + m(x[7], y[0]) + m(x[8], y9_19) + m(x[9], y8_19);
  z[8] = m(x[0], y[8]) + m(x1_2, y[7]) + m(x[2], y[6]) + m(x3_2, y[5]) + m(x[4], y[4]) +
         m(x5_2, y[3]) + m(x[6], y[2]) + m(x7_2, y[1]) + m(x[8], y[0]) + m(x9_2, y9_19);
  z[9] = m(x[0], y[9]) + m(x[1], y[8]) + m(x[2], y[7]) + m(x[3], y[6]) + m(x[4], y[5]) +
         m(x[5], y[4]) + m(x[6], y[3]) + m(x[7], y[2]) + m(x[8], y[1]) + m(x[9], y[0]);

  uint32_t h_tmp[10];
  f32b::reduce(h_tmp, z);

  h[0] = h_tmp[0];
  h[1] = h_tmp[1];
  h[2] = h_tmp[2];
  h[3] = h_tmp[3];
  h[4] = h_tmp[4];
  h[5] = h_tmp[5];
  h[6] = h_tmp[6];
  h[7] = h_tmp[7];
  h[8] = h_tmp[8];
  h[9] = h_tmp[9];
}

//--------------------------------------------------------------------------------------------------
// mul32
//--------------------------------------------------------------------------------------------------
/*
 * I believe this is only used for creating _c21 points, so not time critical.
 */
void mul32(f32t::element& h, const f32t::element& f, uint32_t n) noexcept {
  f32t::element n_elm{n, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  mul(h, f, n_elm);
}
} // namespace sxt::f32o
