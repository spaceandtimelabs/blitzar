/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */
#include "sxt/field51/operation/notsquare.h"

#include "sxt/field51/base/byte_conversion.h"
#include "sxt/field51/operation/mul.h"
#include "sxt/field51/operation/sq.h"
#include "sxt/field51/operation/sqmul.h"
#include "sxt/field51/type/element.h"

namespace sxt::f51o {
//--------------------------------------------------------------------------------------------------
// notsquare
//--------------------------------------------------------------------------------------------------
int notsquare(const f51t::element& x) noexcept {
  f51t::element _10, _11, _1100, _1111, _11110000, _11111111;
  f51t::element t, u, v;
  unsigned char s[32];

  /* Jacobi symbol - x^((p-1)/2) */
  mul(_10, x, x);
  mul(_11, x, _10);
  sq(_1100, _11);
  sq(_1100, _1100);
  mul(_1111, _11, _1100);
  sq(_11110000, _1111);
  sq(_11110000, _11110000);
  sq(_11110000, _11110000);
  sq(_11110000, _11110000);
  mul(_11111111, _1111, _11110000);
  t = _11111111;
  sqmul(t, 2, _11);
  u = t;
  sqmul(t, 10, u);
  sqmul(t, 10, u);
  v = t;
  sqmul(t, 30, v);
  v = t;
  sqmul(t, 60, v);
  v = t;
  sqmul(t, 120, v);
  sqmul(t, 10, u);
  sqmul(t, 3, _11);
  sq(t, t);

  f51b::to_bytes(s, t.data());

  return s[1] & 1;
}
} // namespace sxt::f51o
