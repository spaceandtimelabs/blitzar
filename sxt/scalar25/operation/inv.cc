/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */
#include "sxt/scalar25/operation/inv.h"

#include <cassert>

#include "sxt/scalar25/operation/mul.h"
#include "sxt/scalar25/operation/sq.h"
#include "sxt/scalar25/operation/sqmul.h"
#include "sxt/scalar25/property/zero.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// inv
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void inv(s25t::element& s_inv, const s25t::element& s) noexcept {
  assert(!s25p::is_zero(s));

  s25t::element _10, _100, _1000, _10000, _100000, _1000000, _10010011, _10010111, _100110, _1010,
      _1010000, _1010011, _1011, _10110, _10111101, _11, _1100011, _1100111, _11010011, _1101011,
      _11100111, _11101011, _11110101;

  sq(_10, s);
  mul(_11, s, _10);
  mul(_100, s, _11);
  sq(_1000, _100);
  mul(_1010, _10, _1000);
  mul(_1011, s, _1010);
  sq(_10000, _1000);
  sq(_10110, _1011);
  mul(_100000, _1010, _10110);
  mul(_100110, _10000, _10110);
  sq(_1000000, _100000);
  mul(_1010000, _10000, _1000000);
  mul(_1010011, _11, _1010000);
  mul(_1100011, _10000, _1010011);
  mul(_1100111, _100, _1100011);
  mul(_1101011, _100, _1100111);
  mul(_10010011, _1000000, _1010011);
  mul(_10010111, _100, _10010011);
  mul(_10111101, _100110, _10010111);
  mul(_11010011, _10110, _10111101);
  mul(_11100111, _1010000, _10010111);
  mul(_11101011, _100, _11100111);
  mul(_11110101, _1010, _11101011);

  mul(s_inv, _1011, _11110101);
  sqmul(s_inv, 126, _1010011);
  sqmul(s_inv, 9, _10);
  mul(s_inv, s_inv, _11110101);
  sqmul(s_inv, 7, _1100111);
  sqmul(s_inv, 9, _11110101);
  sqmul(s_inv, 11, _10111101);
  sqmul(s_inv, 8, _11100111);
  sqmul(s_inv, 9, _1101011);
  sqmul(s_inv, 6, _1011);
  sqmul(s_inv, 14, _10010011);
  sqmul(s_inv, 10, _1100011);
  sqmul(s_inv, 9, _10010111);
  sqmul(s_inv, 10, _11110101);
  sqmul(s_inv, 8, _11010011);
  sqmul(s_inv, 8, _11101011);
}

//--------------------------------------------------------------------------------------------------
// batch_inv
//--------------------------------------------------------------------------------------------------
void batch_inv(basct::span<s25t::element> sx_inv, basct::cspan<s25t::element> sx) noexcept {
  // Note: there are more efficient ways to do inversion in bulk; but we're starting
  // with the simplest approach for now.
  assert(sx_inv.size() == sx.size());
  auto n = sx_inv.size();
  for (size_t i = 0; i < n; ++i) {
    inv(sx_inv[i], sx[i]);
  }
}
} // namespace sxt::s25o
