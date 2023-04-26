/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */
#include "sxt/field51/operation/sqrt.h"

#include "sxt/field51/constant/sqrtm1.h"
#include "sxt/field51/operation/cmov.h"
#include "sxt/field51/operation/mul.h"
#include "sxt/field51/operation/pow22523.h"
#include "sxt/field51/operation/sq.h"
#include "sxt/field51/operation/sub.h"
#include "sxt/field51/property/zero.h"
#include "sxt/field51/type/element.h"

namespace sxt::f51o {
//--------------------------------------------------------------------------------------------------
// unchecked_sqrt
//--------------------------------------------------------------------------------------------------
void unchecked_sqrt(f51t::element& x, const f51t::element& x2) noexcept {
  f51t::element p_root;
  f51t::element m_root;
  f51t::element m_root2;
  f51t::element e;

  f51o::pow22523(e, x2);
  f51o::mul(p_root, e, x2);
  f51o::mul(m_root, p_root, f51cn::sqrtm1_v);
  f51o::sq(m_root2, m_root);
  f51o::sub(e, x2, m_root2);
  x = p_root;
  f51o::cmov(x, m_root, f51p::is_zero(e));
}

//--------------------------------------------------------------------------------------------------
// sqrt
//--------------------------------------------------------------------------------------------------
int sqrt(f51t::element& x, const f51t::element& x2) noexcept {
  f51t::element check;
  f51t::element x2_copy;

  x2_copy = x2;
  unchecked_sqrt(x, x2);
  sq(check, x);
  sub(check, check, x2_copy);

  return f51p::is_zero(check) - 1;
}
} // namespace sxt::f51o
