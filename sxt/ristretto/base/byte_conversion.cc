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

#include "sxt/ristretto/base/byte_conversion.h"

#include "sxt/curve32/type/element_p3.h"
#include "sxt/field32/base/byte_conversion.h"
#include "sxt/field32/constant/d.h"
#include "sxt/field32/constant/invsqrtamd.h"
#include "sxt/field32/constant/one.h"
#include "sxt/field32/constant/sqrtm1.h"
#include "sxt/field32/operation/abs.h"
#include "sxt/field32/operation/add.h"
#include "sxt/field32/operation/cmov.h"
#include "sxt/field32/operation/cneg.h"
#include "sxt/field32/operation/mul.h"
#include "sxt/field32/operation/neg.h"
#include "sxt/field32/operation/pow22523.h"
#include "sxt/field32/operation/sq.h"
#include "sxt/field32/operation/sub.h"
#include "sxt/field32/property/sign.h"
#include "sxt/field32/property/zero.h"
#include "sxt/field32/type/element.h"
#include "sxt/ristretto/base/sqrt_ratio_m1.h"

namespace sxt::rstb {
//--------------------------------------------------------------------------------------------------
// is_canonical
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
static int is_canonical(const unsigned char* s) {
  unsigned char c;
  unsigned char d;
  unsigned char e;
  unsigned int i;

  c = (s[31] & 0x7f) ^ 0x7f;
  for (i = 30; i > 0; i--) {
    c |= s[i] ^ 0xff;
  }
  c = (((unsigned int)c) - 1U) >> 8;
  d = (0xed - 1U - (unsigned int)s[0]) >> 8;
  e = s[31] >> 7;

  return 1 - (((c & d) | e | s[0]) & 1);
}

//--------------------------------------------------------------------------------------------------
// to_bytes
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void to_bytes(uint8_t s[32], const c32t::element_p3& p) noexcept {
  f32t::element den1, den2;
  f32t::element den_inv;
  f32t::element eden;
  f32t::element inv_sqrt;
  f32t::element ix, iy;
  f32t::element one{f32cn::one_v};
  f32t::element s_;
  f32t::element t_z_inv;
  f32t::element u1, u2;
  f32t::element u1_u2u2;
  f32t::element x_, y_;
  f32t::element x_z_inv;
  f32t::element z_inv;
  f32t::element zmy;
  int rotate;

  f32o::add(u1, p.Z, p.Y);  /* u1 = Z+Y */
  f32o::sub(zmy, p.Z, p.Y); /* zmy = Z-Y */
  f32o::mul(u1, u1, zmy);   /* u1 = (Z+Y)*(Z-Y) */
  f32o::mul(u2, p.X, p.Y);  /* u2 = X*Y */

  f32o::sq(u1_u2u2, u2);           /* u1_u2u2 = u2^2 */
  f32o::mul(u1_u2u2, u1, u1_u2u2); /* u1_u2u2 = u1*u2^2 */

  (void)rstb::compute_sqrt_ratio_m1(inv_sqrt, one, u1_u2u2);

  f32o::mul(den1, inv_sqrt, u1); /* den1 = inv_sqrt*u1 */
  f32o::mul(den2, inv_sqrt, u2); /* den2 = inv_sqrt*u2 */
  f32o::mul(z_inv, den1, den2);  /* z_inv = den1*den2 */
  f32o::mul(z_inv, z_inv, p.T);  /* z_inv = den1*den2*T */

  f32o::mul(ix, p.X, f32t::element{f32cn::sqrtm1_v});      /* ix = X*sqrt(-1) */
  f32o::mul(iy, p.Y, f32t::element{f32cn::sqrtm1_v});      /* iy = Y*sqrt(-1) */
  f32o::mul(eden, den1, f32t::element{f32cn::invsqrtamd}); /* eden = den1/sqrt(a-d) */

  f32o::mul(t_z_inv, p.T, z_inv); /* t_z_inv = T*z_inv */
  rotate = f32p::is_negative(t_z_inv);

  x_ = p.X;
  y_ = p.Y;
  den_inv = den2;

  f32o::cmov(x_, iy, rotate);
  f32o::cmov(y_, ix, rotate);
  f32o::cmov(den_inv, eden, rotate);

  f32o::mul(x_z_inv, x_, z_inv);
  f32o::cneg(y_, y_, f32p::is_negative(x_z_inv));

  f32o::sub(s_, p.Z, y_);
  f32o::mul(s_, den_inv, s_);
  f32o::abs(s_, s_);

  f32b::to_bytes(s, s_.data());
}

//--------------------------------------------------------------------------------------------------
// from_bytes
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
int from_bytes(c32t::element_p3& p, const uint8_t* s) noexcept {
  f32t::element inv_sqrt;
  f32t::element one;
  f32t::element s_;
  f32t::element ss;
  f32t::element u1, u2;
  f32t::element u1u1, u2u2;
  f32t::element v;
  f32t::element v_u2u2;
  int was_square;

  if (is_canonical(s) == 0) {
    return -1;
  }

  f32b::from_bytes(s_.data(), s);

  f32o::sq(ss, s_); /* ss = s^2 */

  u1 = f32t::element{f32cn::one_v};
  f32o::sub(u1, u1, ss); /* u1 = 1-ss */
  f32o::sq(u1u1, u1);    /* u1u1 = u1^2 */

  u2 = f32t::element{f32cn::one_v};
  f32o::add(u2, u2, ss); /* u2 = 1+ss */
  f32o::sq(u2u2, u2);    /* u2u2 = u2^2 */

  f32o::mul(v, f32t::element{f32cn::d_v}, u1u1); /* v = d*u1^2 */
  f32o::neg(v, v);                               /* v = -d*u1^2 */
  f32o::sub(v, v, u2u2);                         /* v = -(d*u1^2)-u2^2 */

  f32o::mul(v_u2u2, v, u2u2); /* v_u2u2 = v*u2^2 */

  one = f32t::element{f32cn::one_v};
  was_square = rstb::compute_sqrt_ratio_m1(inv_sqrt, one, v_u2u2);

  f32o::mul(p.X, inv_sqrt, u2);
  f32o::mul(p.Y, inv_sqrt, p.X);
  f32o::mul(p.Y, p.Y, v);

  f32o::mul(p.X, p.X, s_);
  f32o::add(p.X, p.X, p.X);
  f32o::abs(p.X, p.X);
  f32o::mul(p.Y, u1, p.Y);

  p.Z = f32t::element{f32cn::one_v};

  f32o::mul(p.T, p.X, p.Y);

  return -((1 - was_square) | f32p::is_negative(p.T) | f32p::is_zero(p.Y));
}
} // namespace sxt::rstb
