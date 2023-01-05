/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */

#include "sxt/ristretto/base/elligator.h"

#include "sxt/curve21/type/element_p3.h"
#include "sxt/field51/constant/d.h"
#include "sxt/field51/constant/one.h"
#include "sxt/field51/constant/sqrtm1.h"
#include "sxt/field51/operation/abs.h"
#include "sxt/field51/operation/add.h"
#include "sxt/field51/operation/cmov.h"
#include "sxt/field51/operation/mul.h"
#include "sxt/field51/operation/neg.h"
#include "sxt/field51/operation/sq.h"
#include "sxt/field51/operation/sub.h"
#include "sxt/field51/type/element.h"
#include "sxt/ristretto/base/sqrt_ratio_m1.h"

namespace sxt::rstb {
//--------------------------------------------------------------------------------------------------
// apply_elligator
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void apply_elligator(c21t::element_p3& p, const f51t::element& t) noexcept {
  f51t::element c;
  f51t::element n;
  f51t::element r;
  f51t::element rpd;
  f51t::element s, s_prime;
  f51t::element ss;
  f51t::element u, v;
  f51t::element w0, w1, w2, w3;
  int wasnt_square;
  auto one = f51cn::one_v;

  f51o::sq(r, t);                                   /* r = t^2 */
  f51o::mul(r, f51t::element{f51cn::sqrtm1_v}, r);  /* r = sqrt(-1)*t^2 */
  f51o::add(u, r, one);                             /* u = r+1 */
  f51o::mul(u, u, f51t::element{f51cn::onemsqd_v}); /* u = (r+1)*(1-d^2) */
  c = f51cn::one_v;
  f51o::neg(c, c);                              /* c = -1 */
  f51o::add(rpd, r, f51t::element{f51cn::d_v}); /* rpd = r+d */
  f51o::mul(v, r, f51t::element{f51cn::d_v});   /* v = r*d */
  f51o::sub(v, c, v);                           /* v = c-r*d */
  f51o::mul(v, v, rpd);                         /* v = (c-r*d)*(r+d) */

  wasnt_square = 1 - rstb::compute_sqrt_ratio_m1(s, u, v);
  f51o::mul(s_prime, s, t);
  f51o::abs(s_prime, s_prime);
  f51o::neg(s_prime, s_prime); /* s_prime = -|s*t| */
  f51o::cmov(s, s_prime, wasnt_square);
  f51o::cmov(c, r, wasnt_square);

  f51o::sub(n, r, one);                             /* n = r-1 */
  f51o::mul(n, n, c);                               /* n = c*(r-1) */
  f51o::mul(n, n, f51t::element{f51cn::sqdmone_v}); /* n = c*(r-1)*(d-1)^2 */
  f51o::sub(n, n, v);                               /* n =  c*(r-1)*(d-1)^2-v */

  f51o::add(w0, s, s);                                /* w0 = 2s */
  f51o::mul(w0, w0, v);                               /* w0 = 2s*v */
  f51o::mul(w1, n, f51t::element{f51cn::sqrtadm1_v}); /* w1 = n*sqrt(ad-1) */
  f51o::sq(ss, s);                                    /* ss = s^2 */
  f51o::sub(w2, one, ss);                             /* w2 = 1-s^2 */
  f51o::add(w3, one, ss);                             /* w3 = 1+s^2 */

  f51o::mul(p.X, w0, w3);
  f51o::mul(p.Y, w2, w1);
  f51o::mul(p.Z, w1, w3);
  f51o::mul(p.T, w0, w2);
}
} // namespace sxt::rstb
