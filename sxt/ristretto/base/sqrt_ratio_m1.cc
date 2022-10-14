/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */

#include "sxt/ristretto/base/sqrt_ratio_m1.h"

#include "sxt/curve21/type/element_p3.h"
#include "sxt/field51/constant/sqrtm1.h"
#include "sxt/field51/operation/abs.h"
#include "sxt/field51/operation/add.h"
#include "sxt/field51/operation/cmov.h"
#include "sxt/field51/operation/mul.h"
#include "sxt/field51/operation/pow22523.h"
#include "sxt/field51/operation/sq.h"
#include "sxt/field51/operation/sub.h"
#include "sxt/field51/property/zero.h"

namespace sxt::rstb {
//--------------------------------------------------------------------------------------------------
// compute_sqrt_ratio_m1
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
int compute_sqrt_ratio_m1(f51t::element& x, const f51t::element& u,
                          const f51t::element& v) noexcept {
  f51t::element v3;
  f51t::element vxx;
  f51t::element m_root_check, p_root_check, f_root_check;
  f51t::element x_sqrtm1;
  int has_m_root, has_p_root, has_f_root;

  f51o::sq(v3, v);
  f51o::mul(v3, v3, v); /* v3 = v^3 */
  f51o::sq(x, v3);
  f51o::mul(x, x, u);
  f51o::mul(x, x, v); /* x = uv^7 */

  f51o::pow22523(x, x); /* x = (uv^7)^((q-5)/8) */
  f51o::mul(x, x, v3);
  f51o::mul(x, x, u); /* x = uv^3(uv^7)^((q-5)/8) */

  f51o::sq(vxx, x);
  f51o::mul(vxx, vxx, v);                                     /* vx^2 */
  f51o::sub(m_root_check, vxx, u);                            /* vx^2-u */
  f51o::add(p_root_check, vxx, u);                            /* vx^2+u */
  f51o::mul(f_root_check, u, f51t::element{f51cn::sqrtm1_v}); /* u*sqrt(-1) */
  f51o::add(f_root_check, vxx, f_root_check);                 /* vx^2+u*sqrt(-1) */
  has_m_root = f51p::is_zero(m_root_check);
  has_p_root = f51p::is_zero(p_root_check);
  has_f_root = f51p::is_zero(f_root_check);
  f51o::mul(x_sqrtm1, x, f51t::element{f51cn::sqrtm1_v}); /* x*sqrt(-1) */

  f51o::cmov(x, x_sqrtm1, has_p_root | has_f_root);
  f51o::abs(x, x);

  return has_m_root | has_p_root;
}
} // namespace sxt::rstb
