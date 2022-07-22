/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */

#pragma once

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/curve21/type/element_cached.h"
#include "sxt/curve21/type/element_p1p1.h"
#include "sxt/curve21/type/element_p2.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/field51/constant/d.h"
#include "sxt/field51/operation/add.h"
#include "sxt/field51/operation/mul.h"
#include "sxt/field51/operation/sub.h"

namespace sxt::c21t {
struct element_p3;
struct element_cached;

//--------------------------------------------------------------------------------------------------
// to_element_cached
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
inline void to_element_cached(element_cached& r, const element_p3& p) noexcept {
  f51o::add(r.YplusX, p.Y, p.X);
  f51o::sub(r.YminusX, p.Y, p.X);
  r.Z = p.Z;
  f51o::mul(r.T2d, p.T, f51t::element{f51cn::d2_v});
}

//--------------------------------------------------------------------------------------------------
// to_element_p2
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
inline void to_element_p2(element_p2& r, const element_p3& p) noexcept {
  r.X = p.X;
  r.Y = p.Y;
  r.Z = p.Z;
}

CUDA_CALLABLE
inline void to_element_p2(element_p2& r, const element_p1p1& p) noexcept {
  f51o::mul(r.X, p.X, p.T);
  f51o::mul(r.Y, p.Y, p.Z);
  f51o::mul(r.Z, p.Z, p.T);
}

//--------------------------------------------------------------------------------------------------
// to_element_p3
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
inline void to_element_p3(element_p3& r, const element_p1p1& p) noexcept {
  f51o::mul(r.X, p.X, p.T);
  f51o::mul(r.Y, p.Y, p.Z);
  f51o::mul(r.Z, p.Z, p.T);
  f51o::mul(r.T, p.X, p.Y);
}
} // namespace sxt::c21t
