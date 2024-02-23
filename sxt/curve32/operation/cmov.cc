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
#include "sxt/curve32/operation/cmov.h"

#include "sxt/curve32/constant/identity.h"
#include "sxt/curve32/type/element_cached.h"
#include "sxt/field32/operation/cmov.h"
#include "sxt/field32/operation/neg.h"

namespace sxt::c32o {
//--------------------------------------------------------------------------------------------------
// negative
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
static inline unsigned char negative(signed char b) noexcept {
  /* 18446744073709551361..18446744073709551615: yes; 0..255: no */
  uint64_t x = b;

  x >>= 63; /* 1: yes; 0: no */

  return x;
}

//--------------------------------------------------------------------------------------------------
// equal
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
static inline unsigned char equal(signed char b, signed char c) noexcept {
  unsigned char ub = b;
  unsigned char uc = c;
  unsigned char x = ub ^ uc; /* 0: yes; 1..255: no */
  uint32_t y = (uint32_t)x;  /* 0: yes; 1..255: no */

  y -= 1;   /* 4294967295: yes; 0..254: no */
  y >>= 31; /* 1: yes; 0: no */

  return y;
}

//--------------------------------------------------------------------------------------------------
// cmov
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void cmov(c32t::element_cached& t, const c32t::element_cached& u, unsigned char b) noexcept {
  f32o::cmov(t.YplusX, u.YplusX, b);
  f32o::cmov(t.YminusX, u.YminusX, b);
  f32o::cmov(t.Z, u.Z, b);
  f32o::cmov(t.T2d, u.T2d, b);
}

//--------------------------------------------------------------------------------------------------
// cmov8
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE void cmov8(c32t::element_cached& t, const c32t::element_cached cached[8],
                         const signed char b) noexcept {
  c32t::element_cached minust;
  const unsigned char bnegative = negative(b);
  const unsigned char babs = b - (((-bnegative) & b) * ((signed char)1 << 1));

  t = c32cn::identity_cached_v;
  cmov(t, cached[0], equal(babs, 1));
  cmov(t, cached[1], equal(babs, 2));
  cmov(t, cached[2], equal(babs, 3));
  cmov(t, cached[3], equal(babs, 4));
  cmov(t, cached[4], equal(babs, 5));
  cmov(t, cached[5], equal(babs, 6));
  cmov(t, cached[6], equal(babs, 7));
  cmov(t, cached[7], equal(babs, 8));

  minust.YplusX = t.YminusX;
  minust.YminusX = t.YplusX;
  minust.Z = t.Z;
  f32o::neg(minust.T2d, t.T2d);
  cmov(t, minust, bnegative);
}
} // namespace sxt::c32o
