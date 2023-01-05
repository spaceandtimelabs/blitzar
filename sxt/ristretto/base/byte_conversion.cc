/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */

#include "sxt/ristretto/base/byte_conversion.h"

#include "sxt/curve21/type/element_p3.h"
#include "sxt/field51/base/byte_conversion.h"
#include "sxt/field51/constant/d.h"
#include "sxt/field51/constant/invsqrtamd.h"
#include "sxt/field51/constant/one.h"
#include "sxt/field51/constant/sqrtm1.h"
#include "sxt/field51/operation/abs.h"
#include "sxt/field51/operation/add.h"
#include "sxt/field51/operation/cmov.h"
#include "sxt/field51/operation/cneg.h"
#include "sxt/field51/operation/mul.h"
#include "sxt/field51/operation/neg.h"
#include "sxt/field51/operation/pow22523.h"
#include "sxt/field51/operation/sq.h"
#include "sxt/field51/operation/sub.h"
#include "sxt/field51/property/sign.h"
#include "sxt/field51/property/zero.h"
#include "sxt/field51/type/element.h"
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
void to_bytes(uint8_t s[32], const c21t::element_p3& p) noexcept {
  f51t::element den1, den2;
  f51t::element den_inv;
  f51t::element eden;
  f51t::element inv_sqrt;
  f51t::element ix, iy;
  f51t::element one{f51cn::one_v};
  f51t::element s_;
  f51t::element t_z_inv;
  f51t::element u1, u2;
  f51t::element u1_u2u2;
  f51t::element x_, y_;
  f51t::element x_z_inv;
  f51t::element z_inv;
  f51t::element zmy;
  int rotate;

  f51o::add(u1, p.Z, p.Y);  /* u1 = Z+Y */
  f51o::sub(zmy, p.Z, p.Y); /* zmy = Z-Y */
  f51o::mul(u1, u1, zmy);   /* u1 = (Z+Y)*(Z-Y) */
  f51o::mul(u2, p.X, p.Y);  /* u2 = X*Y */

  f51o::sq(u1_u2u2, u2);           /* u1_u2u2 = u2^2 */
  f51o::mul(u1_u2u2, u1, u1_u2u2); /* u1_u2u2 = u1*u2^2 */

  (void)rstb::compute_sqrt_ratio_m1(inv_sqrt, one, u1_u2u2);

  f51o::mul(den1, inv_sqrt, u1); /* den1 = inv_sqrt*u1 */
  f51o::mul(den2, inv_sqrt, u2); /* den2 = inv_sqrt*u2 */
  f51o::mul(z_inv, den1, den2);  /* z_inv = den1*den2 */
  f51o::mul(z_inv, z_inv, p.T);  /* z_inv = den1*den2*T */

  f51o::mul(ix, p.X, f51t::element{f51cn::sqrtm1_v});      /* ix = X*sqrt(-1) */
  f51o::mul(iy, p.Y, f51t::element{f51cn::sqrtm1_v});      /* iy = Y*sqrt(-1) */
  f51o::mul(eden, den1, f51t::element{f51cn::invsqrtamd}); /* eden = den1/sqrt(a-d) */

  f51o::mul(t_z_inv, p.T, z_inv); /* t_z_inv = T*z_inv */
  rotate = f51p::is_negative(t_z_inv);

  x_ = p.X;
  y_ = p.Y;
  den_inv = den2;

  f51o::cmov(x_, iy, rotate);
  f51o::cmov(y_, ix, rotate);
  f51o::cmov(den_inv, eden, rotate);

  f51o::mul(x_z_inv, x_, z_inv);
  f51o::cneg(y_, y_, f51p::is_negative(x_z_inv));

  f51o::sub(s_, p.Z, y_);
  f51o::mul(s_, den_inv, s_);
  f51o::abs(s_, s_);

  f51b::to_bytes(s, s_.data());
}

//--------------------------------------------------------------------------------------------------
// from_bytes
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
int from_bytes(c21t::element_p3& p, const uint8_t* s) noexcept {
  f51t::element inv_sqrt;
  f51t::element one;
  f51t::element s_;
  f51t::element ss;
  f51t::element u1, u2;
  f51t::element u1u1, u2u2;
  f51t::element v;
  f51t::element v_u2u2;
  int was_square;

  if (is_canonical(s) == 0) {
    return -1;
  }

  f51b::from_bytes(s_.data(), s);

  f51o::sq(ss, s_); /* ss = s^2 */

  u1 = f51t::element{f51cn::one_v};
  f51o::sub(u1, u1, ss); /* u1 = 1-ss */
  f51o::sq(u1u1, u1);    /* u1u1 = u1^2 */

  u2 = f51t::element{f51cn::one_v};
  f51o::add(u2, u2, ss); /* u2 = 1+ss */
  f51o::sq(u2u2, u2);    /* u2u2 = u2^2 */

  f51o::mul(v, f51t::element{f51cn::d_v}, u1u1); /* v = d*u1^2 */
  f51o::neg(v, v);                               /* v = -d*u1^2 */
  f51o::sub(v, v, u2u2);                         /* v = -(d*u1^2)-u2^2 */

  f51o::mul(v_u2u2, v, u2u2); /* v_u2u2 = v*u2^2 */

  one = f51t::element{f51cn::one_v};
  was_square = rstb::compute_sqrt_ratio_m1(inv_sqrt, one, v_u2u2);

  f51o::mul(p.X, inv_sqrt, u2);
  f51o::mul(p.Y, inv_sqrt, p.X);
  f51o::mul(p.Y, p.Y, v);

  f51o::mul(p.X, p.X, s_);
  f51o::add(p.X, p.X, p.X);
  f51o::abs(p.X, p.X);
  f51o::mul(p.Y, u1, p.Y);

  p.Z = f51t::element{f51cn::one_v};

  f51o::mul(p.T, p.X, p.Y);

  return -((1 - was_square) | f51p::is_negative(p.T) | f51p::is_zero(p.Y));
}
} // namespace sxt::rstb
