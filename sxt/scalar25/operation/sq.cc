/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */
#include "sxt/base/bit/load.h"
#include "sxt/scalar25/operation/mul.h"

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// sq
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void sq(s25t::element& s, const s25t::element& a) noexcept {
  // adopted from libsodium's sc25519_sq
  mul(s, a, a);
}
} // namespace sxt::s25o
