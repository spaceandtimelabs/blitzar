/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */
#include "sxt/scalar25/operation/sqmul.h"

#include "sxt/scalar25/operation/mul.h"
#include "sxt/scalar25/operation/sq.h"

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// sqmul
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void sqmul(s25t::element& s, const uint32_t n, const s25t::element& a) noexcept {
  // adopted from libsodium's sc25519_sqmul
  for (uint32_t i = 0; i < n; i++) {
    sq(s, s);
  }

  mul(s, s, a);
}
} // namespace sxt::s25o
