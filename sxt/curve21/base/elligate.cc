/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */
#include "sxt/curve21/base/elligate.h"

#include "sxt/base/error/panic.h"
#include "sxt/field51/constant/zero.h"
#include "sxt/field51/operation/add.h"
#include "sxt/field51/operation/cmov.h"
#include "sxt/field51/operation/invert.h"
#include "sxt/field51/operation/mul.h"
#include "sxt/field51/operation/neg.h"
#include "sxt/field51/operation/notsquare.h"
#include "sxt/field51/operation/sq.h"
#include "sxt/field51/operation/sqrt.h"
#include "sxt/field51/type/element.h"

namespace sxt::c21b {
static constexpr uint32_t ed25519_A_32 = 486662;
static constexpr f51t::element ed25519_A = {ed25519_A_32, 0, 0, 0, 0};

//--------------------------------------------------------------------------------------------------
// xmont_to_ymont
//--------------------------------------------------------------------------------------------------
static int xmont_to_ymont(f51t::element& y, const f51t::element& x) noexcept {
  f51t::element x2;
  f51t::element x3;

  f51o::sq(x2, x);
  f51o::mul(x3, x, x2);
  f51o::mul32(x2, x2, ed25519_A_32);
  f51o::add(y, x3, x);
  f51o::add(y, y, x2);

  return f51o::sqrt(y, y);
}

//--------------------------------------------------------------------------------------------------
// apply_elligator
//--------------------------------------------------------------------------------------------------
void apply_elligator(f51t::element& x, f51t::element& y, int* notsquare_p,
                     const f51t::element& r) noexcept {
  f51t::element gx1;
  f51t::element rr2;
  f51t::element x2, x3, negx;
  int notsquare;

  f51o::sq2(rr2, r);
  rr2[0]++;
  f51o::invert(rr2, rr2);
  f51o::mul32(x, rr2, ed25519_A_32);
  f51o::neg(x, x); /* x=x1 */

  f51o::sq(x2, x);
  f51o::mul(x3, x, x2);
  f51o::mul32(x2, x2, ed25519_A_32); /* x2 = A*x1^2 */
  f51o::add(gx1, x3, x);
  f51o::add(gx1, gx1, x2); /* gx1 = x1^3 + A*x1^2 + x1 */

  notsquare = f51o::notsquare(gx1);

  /* gx1 not a square  => x = -x1-A */
  f51o::neg(negx, x);
  f51o::cmov(x, negx, notsquare);
  x2 = f51cn::zero_v;
  f51o::cmov(x2, ed25519_A, notsquare);
  f51o::sub(x, x, x2);

  /* y = sqrt(gx1) or sqrt(gx2) with gx2 = gx1 * (A+x1) / -x1 */
  /* but it is about as fast to just recompute from the curve equation. */
  if (xmont_to_ymont(y, x) != 0) {
    baser::panic("root should exist");
  }
  *notsquare_p = notsquare;
}
} // namespace sxt::c21b
