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
#include "sxt/curve32/base/elligate.h"

#include "sxt/base/error/panic.h"
#include "sxt/field32/constant/zero.h"
#include "sxt/field32/operation/add.h"
#include "sxt/field32/operation/cmov.h"
#include "sxt/field32/operation/invert.h"
#include "sxt/field32/operation/mul.h"
#include "sxt/field32/operation/neg.h"
#include "sxt/field32/operation/notsquare.h"
#include "sxt/field32/operation/sq.h"
#include "sxt/field32/operation/sqrt.h"
#include "sxt/field32/type/element.h"

namespace sxt::c32b {
static constexpr uint32_t ed25519_A_32 = 486662;
static constexpr f32t::element ed25519_A = {ed25519_A_32, 0, 0, 0, 0, 0, 0, 0, 0, 0};

//--------------------------------------------------------------------------------------------------
// xmont_to_ymont
//--------------------------------------------------------------------------------------------------
static int xmont_to_ymont(f32t::element& y, const f32t::element& x) noexcept {
  f32t::element x2;
  f32t::element x3;

  f32o::sq(x2, x);
  f32o::mul(x3, x, x2);
  f32o::mul32(x2, x2, ed25519_A_32);
  f32o::add(y, x3, x);
  f32o::add(y, y, x2);

  return f32o::sqrt(y, y);
}

//--------------------------------------------------------------------------------------------------
// apply_elligator
//--------------------------------------------------------------------------------------------------
void apply_elligator(f32t::element& x, f32t::element& y, int* notsquare_p,
                     const f32t::element& r) noexcept {
  f32t::element gx1;
  f32t::element rr2;
  f32t::element x2, x3, negx;
  int notsquare;

  f32o::sq2(rr2, r);
  rr2[0]++;
  f32o::invert(rr2, rr2);
  f32o::mul32(x, rr2, ed25519_A_32);
  f32o::neg(x, x); /* x=x1 */

  f32o::sq(x2, x);
  f32o::mul(x3, x, x2);
  f32o::mul32(x2, x2, ed25519_A_32); /* x2 = A*x1^2 */
  f32o::add(gx1, x3, x);
  f32o::add(gx1, gx1, x2); /* gx1 = x1^3 + A*x1^2 + x1 */

  notsquare = f32o::notsquare(gx1);

  /* gx1 not a square  => x = -x1-A */
  f32o::neg(negx, x);
  f32o::cmov(x, negx, notsquare);
  x2 = f32cn::zero_v;
  f32o::cmov(x2, ed25519_A, notsquare);
  f32o::sub(x, x, x2);

  /* y = sqrt(gx1) or sqrt(gx2) with gx2 = gx1 * (A+x1) / -x1 */
  /* but it is about as fast to just recompute from the curve equation. */
  if (xmont_to_ymont(y, x) != 0) {
    baser::panic("root should exist");
  }
  *notsquare_p = notsquare;
}
} // namespace sxt::c32b
