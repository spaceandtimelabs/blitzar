/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */

#include "sxt/field51/operation/mul.h"

#include "sxt/base/type/int.h"
#include "sxt/field51/type/element.h"

namespace sxt::f51o {
//--------------------------------------------------------------------------------------------------
// mul_impl
//--------------------------------------------------------------------------------------------------
template <class T>
CUDA_CALLABLE static void mul_impl(T& h, const f51t::element& f, const f51t::element& g) noexcept {
  const uint64_t mask = 0x7ffffffffffffULL;
  uint128_t r0, r1, r2, r3, r4;
  uint128_t f0, f1, f2, f3, f4;
  uint128_t f1_19, f2_19, f3_19, f4_19;
  uint128_t g0, g1, g2, g3, g4;
  uint64_t r00, r01, r02, r03, r04;
  uint64_t carry;

  f0 = (uint128_t)f[0];
  f1 = (uint128_t)f[1];
  f2 = (uint128_t)f[2];
  f3 = (uint128_t)f[3];
  f4 = (uint128_t)f[4];

  g0 = (uint128_t)g[0];
  g1 = (uint128_t)g[1];
  g2 = (uint128_t)g[2];
  g3 = (uint128_t)g[3];
  g4 = (uint128_t)g[4];

  f1_19 = 19ULL * f1;
  f2_19 = 19ULL * f2;
  f3_19 = 19ULL * f3;
  f4_19 = 19ULL * f4;

  r0 = f0 * g0 + f1_19 * g4 + f2_19 * g3 + f3_19 * g2 + f4_19 * g1;
  r1 = f0 * g1 + f1 * g0 + f2_19 * g4 + f3_19 * g3 + f4_19 * g2;
  r2 = f0 * g2 + f1 * g1 + f2 * g0 + f3_19 * g4 + f4_19 * g3;
  r3 = f0 * g3 + f1 * g2 + f2 * g1 + f3 * g0 + f4_19 * g4;
  r4 = f0 * g4 + f1 * g3 + f2 * g2 + f3 * g1 + f4 * g0;

  r00 = ((uint64_t)r0) & mask;
  carry = (uint64_t)(r0 >> 51);
  r1 += carry;
  r01 = ((uint64_t)r1) & mask;
  carry = (uint64_t)(r1 >> 51);
  r2 += carry;
  r02 = ((uint64_t)r2) & mask;
  carry = (uint64_t)(r2 >> 51);
  r3 += carry;
  r03 = ((uint64_t)r3) & mask;
  carry = (uint64_t)(r3 >> 51);
  r4 += carry;
  r04 = ((uint64_t)r4) & mask;
  carry = (uint64_t)(r4 >> 51);
  r00 += 19ULL * carry;
  carry = r00 >> 51;
  r00 &= mask;
  r01 += carry;
  carry = r01 >> 51;
  r01 &= mask;
  r02 += carry;

  h[0] = r00;
  h[1] = r01;
  h[2] = r02;
  h[3] = r03;
  h[4] = r04;
}

//--------------------------------------------------------------------------------------------------
// mul
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void mul(f51t::element& h, const f51t::element& f, const f51t::element& g) noexcept {
  mul_impl(h, f, g);
}

CUDA_CALLABLE
void mul(volatile f51t::element& h, const f51t::element& f, const f51t::element& g) noexcept {
  mul_impl(h, f, g);
}

//--------------------------------------------------------------------------------------------------
// mul32
//--------------------------------------------------------------------------------------------------
void mul32(f51t::element& h, const f51t::element& f, uint32_t n) noexcept {
  constexpr uint64_t mask = 0x7ffffffffffffULL;
  uint128_t a;
  uint128_t sn = (uint128_t)n;
  uint64_t h0, h1, h2, h3, h4;

  a = f[0] * sn;
  h0 = ((uint64_t)a) & mask;
  a = f[1] * sn + ((uint64_t)(a >> 51));
  h1 = ((uint64_t)a) & mask;
  a = f[2] * sn + ((uint64_t)(a >> 51));
  h2 = ((uint64_t)a) & mask;
  a = f[3] * sn + ((uint64_t)(a >> 51));
  h3 = ((uint64_t)a) & mask;
  a = f[4] * sn + ((uint64_t)(a >> 51));
  h4 = ((uint64_t)a) & mask;

  h0 += (a >> 51) * 19ULL;

  h[0] = h0;
  h[1] = h1;
  h[2] = h2;
  h[3] = h3;
  h[4] = h4;
}
} // namespace sxt::f51o
