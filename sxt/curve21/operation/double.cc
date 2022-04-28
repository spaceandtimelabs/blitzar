/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */

#include "sxt/curve21/operation/double.h"

#include "sxt/field51/type/element.h"
#include "sxt/field51/operation/square.h"
#include "sxt/field51/operation/add.h"
#include "sxt/field51/operation/sub.h"

namespace sxt::c21o {
//--------------------------------------------------------------------------------------------------
// double_element
//--------------------------------------------------------------------------------------------------
/*
 r = 2 * p
*/
CUDA_CALLABLE
void double_element(c21t::element_p1p1 &r, const c21t::element_p2 &p) noexcept {
  f51t::element t0;

  f51o::square(r.X, p.X);
  f51o::square(r.Z, p.Y);
  f51o::square2(r.T, p.Z);
  f51o::add(r.Y, p.X, p.Y);
  f51o::square(t0, r.Y);
  f51o::add(r.Y, r.Z, r.X);
  f51o::sub(r.Z, r.Z, r.X);
  f51o::sub(r.X, t0, r.Y);
  f51o::sub(r.T, r.T, r.Z);
}
} // namespace sxt::c21o
