/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */

#include "sxt/curve21/operation/neg.h"

#include "sxt/curve21/type/element_p3.h"
#include "sxt/field51/operation/neg.h"

namespace sxt::c21o {
//--------------------------------------------------------------------------------------------------
// neg
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void neg(c21t::element_p3& r, const c21t::element_p3& p) noexcept {
  f51o::neg(r.X, p.X);
  r.Y = p.Y;
  r.Z = p.Z;
  f51o::neg(r.T, p.T);
}
} // namespace sxt::c21o
