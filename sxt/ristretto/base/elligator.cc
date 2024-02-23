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

#include "sxt/ristretto/base/elligator.h"

#include "sxt/curve32/type/element_p3.h"
#include "sxt/field32/constant/d.h"
#include "sxt/field32/constant/one.h"
#include "sxt/field32/constant/sqrtm1.h"
#include "sxt/field32/operation/abs.h"
#include "sxt/field32/operation/add.h"
#include "sxt/field32/operation/cmov.h"
#include "sxt/field32/operation/mul.h"
#include "sxt/field32/operation/neg.h"
#include "sxt/field32/operation/sq.h"
#include "sxt/field32/operation/sub.h"
#include "sxt/field32/type/element.h"
#include "sxt/ristretto/base/sqrt_ratio_m1.h"

namespace sxt::rstb {
//--------------------------------------------------------------------------------------------------
// apply_elligator
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void apply_elligator(c32t::element_p3& p, const f32t::element& t) noexcept {
  f32t::element c;
  f32t::element n;
  f32t::element r;
  f32t::element rpd;
  f32t::element s, s_prime;
  f32t::element ss;
  f32t::element u, v;
  f32t::element w0, w1, w2, w3;
  int wasnt_square;
  auto one = f32cn::one_v;

  f32o::sq(r, t);                                   /* r = t^2 */
  f32o::mul(r, f32t::element{f32cn::sqrtm1_v}, r);  /* r = sqrt(-1)*t^2 */
  f32o::add(u, r, one);                             /* u = r+1 */
  f32o::mul(u, u, f32t::element{f32cn::onemsqd_v}); /* u = (r+1)*(1-d^2) */
  c = f32cn::one_v;
  f32o::neg(c, c);                              /* c = -1 */
  f32o::add(rpd, r, f32t::element{f32cn::d_v}); /* rpd = r+d */
  f32o::mul(v, r, f32t::element{f32cn::d_v});   /* v = r*d */
  f32o::sub(v, c, v);                           /* v = c-r*d */
  f32o::mul(v, v, rpd);                         /* v = (c-r*d)*(r+d) */

  wasnt_square = 1 - rstb::compute_sqrt_ratio_m1(s, u, v);
  f32o::mul(s_prime, s, t);
  f32o::abs(s_prime, s_prime);
  f32o::neg(s_prime, s_prime); /* s_prime = -|s*t| */
  f32o::cmov(s, s_prime, wasnt_square);
  f32o::cmov(c, r, wasnt_square);

  f32o::sub(n, r, one);                             /* n = r-1 */
  f32o::mul(n, n, c);                               /* n = c*(r-1) */
  f32o::mul(n, n, f32t::element{f32cn::sqdmone_v}); /* n = c*(r-1)*(d-1)^2 */
  f32o::sub(n, n, v);                               /* n =  c*(r-1)*(d-1)^2-v */

  f32o::add(w0, s, s);                                /* w0 = 2s */
  f32o::mul(w0, w0, v);                               /* w0 = 2s*v */
  f32o::mul(w1, n, f32t::element{f32cn::sqrtadm1_v}); /* w1 = n*sqrt(ad-1) */
  f32o::sq(ss, s);                                    /* ss = s^2 */
  f32o::sub(w2, one, ss);                             /* w2 = 1-s^2 */
  f32o::add(w3, one, ss);                             /* w3 = 1+s^2 */

  f32o::mul(p.X, w0, w3);
  f32o::mul(p.Y, w2, w1);
  f32o::mul(p.Z, w1, w3);
  f32o::mul(p.T, w0, w2);
}
} // namespace sxt::rstb
