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

#include "sxt/field32/operation/sq.h"

#include "sxt/base/type/int.h"
#include "sxt/field32/base/reduce.h"
#include "sxt/field32/type/element.h"

namespace sxt::f32o {
//--------------------------------------------------------------------------------------------------
// sq_inner
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void sq_inner(uint64_t h[10], const f32t::element& x) noexcept {
  // Optimized version of multiplication for the case of squaring.
  // Pre- and post- conditions identical to multiplication function.
  uint32_t x0_2  =  2 * x[0];
  uint32_t x1_2  =  2 * x[1];
  uint32_t x2_2  =  2 * x[2];
  uint32_t x3_2  =  2 * x[3];
  uint32_t x4_2  =  2 * x[4];
  uint32_t x5_2  =  2 * x[5];
  uint32_t x6_2  =  2 * x[6];
  uint32_t x7_2  =  2 * x[7];
  uint32_t x5_19 = 19 * x[5];
  uint32_t x6_19 = 19 * x[6];
  uint32_t x7_19 = 19 * x[7];
  uint32_t x8_19 = 19 * x[8];
  uint32_t x9_19 = 19 * x[9];

  // Helper function to multiply two 32-bit integers with 64 bits
  // of output.
  auto m = [](uint32_t x, uint32_t y) -> uint64_t {
    return static_cast<uint64_t>(x) * static_cast<uint64_t>(y);
  };

  // This block is rearranged so that instead of doing a 32-bit multiplication by 38, we do a
  // 64-bit multiplication by 2 on the results.  This is because lg(38) is too big: we would
  // have less than 1 bit of headroom left, which is too little.
  h[0] = m(x[0], x[0]) + m(x2_2, x8_19) + m(x4_2, x6_19) + (m(x1_2, x9_19) +  m(x3_2, x7_19) + m(x[5], x5_19)) * 2;
  h[1] = m(x0_2, x[1]) + m(x3_2, x8_19) + m(x5_2, x6_19) + (m(x[2], x9_19) +  m(x[4], x7_19)                 ) * 2;
  h[2] = m(x0_2, x[2]) + m(x1_2,  x[1]) + m(x4_2, x8_19) +  m(x[6], x6_19) + (m(x3_2, x9_19) + m(x5_2, x7_19)) * 2;
  h[3] = m(x0_2, x[3]) + m(x1_2,  x[2]) + m(x5_2, x8_19) + (m(x[4], x9_19) +  m(x[6], x7_19)                 ) * 2;
  h[4] = m(x0_2, x[4]) + m(x1_2,  x3_2) + m(x[2],  x[2]) +  m(x6_2, x8_19) + (m(x5_2, x9_19) + m(x[7], x7_19)) * 2;
  h[5] = m(x0_2, x[5]) + m(x1_2,  x[4]) + m(x2_2,  x[3]) +  m(x7_2, x8_19) +  m(x[6], x9_19)                   * 2;
  h[6] = m(x0_2, x[6]) + m(x1_2,  x5_2) + m(x2_2,  x[4]) +  m(x3_2,  x[3]) +  m(x[8], x8_19) + m(x7_2, x9_19)  * 2;
  h[7] = m(x0_2, x[7]) + m(x1_2,  x[6]) + m(x2_2,  x[5]) +  m(x3_2,  x[4]) +  m(x[8], x9_19)                   * 2;
  h[8] = m(x0_2, x[8]) + m(x1_2,  x7_2) + m(x2_2,  x[6]) +  m(x3_2,  x5_2) +  m(x[4],  x[4]) + m(x[9], x9_19)  * 2;
  h[9] = m(x0_2, x[9]) + m(x1_2,  x[8]) + m(x2_2,  x[7]) +  m(x3_2,  x[6]) +  m(x4_2,  x[5])                      ;  
}

//--------------------------------------------------------------------------------------------------
// sq
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void sq(f32t::element& h, const f32t::element& f) noexcept {
  uint64_t t[10] = {0};
  sq_inner(t, f);
  f32b::reduce(h.data(), t);
}

//--------------------------------------------------------------------------------------------------
// sq2
//--------------------------------------------------------------------------------------------------
/*
 h = 2 * f * f
 Can overlap h with f.
*/
CUDA_CALLABLE
void sq2(f32t::element& h, const f32t::element& f) noexcept {
  uint64_t t[10] = {0};
  sq_inner(t, f);
  for (size_t i = 0; i < f.num_limbs_v; ++i) {
    t[i] = t[i] + t[i];
  }
  f32b::reduce(h.data(), t);
}
} // namespace sxt::f32o
