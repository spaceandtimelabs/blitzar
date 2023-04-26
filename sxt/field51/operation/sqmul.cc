/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */
#include "sxt/field51/operation/sqmul.h"

#include "sxt/field51/operation/mul.h"
#include "sxt/field51/operation/sq.h"

namespace sxt::f51o {
//--------------------------------------------------------------------------------------------------
// sqmul
//--------------------------------------------------------------------------------------------------
void sqmul(f51t::element& s, int n, const f51t::element& a) noexcept {
  for (int i = 0; i < n; i++) {
    f51o::sq(s, s);
  }
  f51o::mul(s, s, a);
}
} // namespace sxt::f51o
