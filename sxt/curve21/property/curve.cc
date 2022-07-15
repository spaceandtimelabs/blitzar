/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */

#include "sxt/curve21/property/curve.h"

#include "sxt/curve21/type/element_p3.h"

#include "sxt/field51/constant/d.h"
#include "sxt/field51/operation/add.h"
#include "sxt/field51/operation/mul.h"
#include "sxt/field51/operation/square.h"
#include "sxt/field51/operation/sub.h"
#include "sxt/field51/property/zero.h"
#include "sxt/field51/type/element.h"

namespace sxt::c21p {
//--------------------------------------------------------------------------------------------------
// is_on_curve
//--------------------------------------------------------------------------------------------------
bool is_on_curve(const c21t::element_p3& p) noexcept {
  f51t::element x2;
  f51t::element y2;
  f51t::element z2;
  f51t::element z4;
  f51t::element t0;
  f51t::element t1;

  f51o::square(x2, p.X);
  f51o::square(y2, p.Y);
  f51o::square(z2, p.Z);
  f51o::sub(t0, y2, x2);
  f51o::mul(t0, t0, z2);

  f51o::mul(t1, x2, y2);
  f51o::mul(t1, t1, f51t::element{f51cn::d_v});
  f51o::square(z4, z2);
  f51o::add(t1, t1, z4);
  f51o::sub(t0, t0, t1);

  return f51p::is_zero(t0);
}
} // namespace sxt::c21p
