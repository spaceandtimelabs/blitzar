/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */
#include "sxt/scalar25/operation/sub.h"

#include "sxt/scalar25/operation/add.h"
#include "sxt/scalar25/operation/neg.h"

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// sub
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void sub(s25t::element& s, const s25t::element& a, const s25t::element& b) noexcept {
  s25t::element yn;

  neg(yn, b);
  add(s, a, yn);
}
} // namespace sxt::s25o
