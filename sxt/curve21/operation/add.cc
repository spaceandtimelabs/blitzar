/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */

#include "sxt/curve21/operation/add.h"

#include "sxt/field51/operation/add.h"
#include "sxt/field51/operation/mul.h"
#include "sxt/field51/operation/sub.h"
#include "sxt/field51/type/element.h"

namespace sxt::c21o {
//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
/*
 r = p + q
 */
CUDA_CALLABLE
void add(c21t::element_p1p1& r, const c21t::element_p3& p, const c21t::element_cached& q) noexcept {
  f51t::element t0;

  f51o::add(r.X, p.Y, p.X);
  f51o::sub(r.Y, p.Y, p.X);
  f51o::mul(r.Z, r.X, q.YplusX);
  f51o::mul(r.Y, r.Y, q.YminusX);
  f51o::mul(r.T, q.T2d, p.T);
  f51o::mul(r.X, p.Z, q.Z);
  f51o::add(t0, r.X, r.X);
  f51o::sub(r.X, r.Z, r.Y);
  f51o::add(r.Y, r.Z, r.Y);
  f51o::add(r.Z, t0, r.T);
  f51o::sub(r.T, t0, r.T);
}
} // namespace sxt::c21o
