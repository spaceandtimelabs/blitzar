/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2023
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */
#include "sxt/curve32/type/byte_conversion.h"

#include "sxt/curve32/type/element_p3.h"
#include "sxt/field32/base/byte_conversion.h"
#include "sxt/field32/constant/d.h"
#include "sxt/field32/constant/one.h"
#include "sxt/field32/constant/sqrtm1.h"
#include "sxt/field32/operation/add.h"
#include "sxt/field32/operation/cmov.h"
#include "sxt/field32/operation/invert.h"
#include "sxt/field32/operation/mul.h"
#include "sxt/field32/operation/neg.h"
#include "sxt/field32/operation/pow22523.h"
#include "sxt/field32/operation/sq.h"
#include "sxt/field32/operation/sub.h"
#include "sxt/field32/property/sign.h"
#include "sxt/field32/property/zero.h"
#include "sxt/field32/type/element.h"

namespace sxt::c32t {
//--------------------------------------------------------------------------------------------------
// to_bytes
//--------------------------------------------------------------------------------------------------
void to_bytes(uint8_t s[32], const element_p3& h) noexcept {
  f32t::element recip;
  f32t::element x;
  f32t::element y;

  f32o::invert(recip, h.Z);
  f32o::mul(x, h.X, recip);
  f32o::mul(y, h.Y, recip);
  f32b::to_bytes(s, y.data());
  s[31] ^= f32p::is_negative(x) << 7;
}

//--------------------------------------------------------------------------------------------------
// from_bytes
//--------------------------------------------------------------------------------------------------
int from_bytes(element_p3& h, uint8_t s[32]) noexcept {
  f32t::element u;
  f32t::element v;
  f32t::element vxx;
  f32t::element m_root_check, p_root_check;
  f32t::element negx;
  f32t::element x_sqrtm1;
  int has_m_root, has_p_root;

  f32b::from_bytes(h.Y.data(), s);
  h.Z = f32cn::one_v;
  f32o::sq(u, h.Y);
  f32o::mul(v, u, f32cn::d_v);
  f32o::sub(u, u, h.Z); /* u = y^2-1 */
  f32o::add(v, v, h.Z); /* v = dy^2+1 */

  f32o::mul(h.X, u, v);
  f32o::pow22523(h.X, h.X);
  f32o::mul(h.X, u, h.X); /* u((uv)^((q-5)/8)) */

  f32o::sq(vxx, h.X);
  f32o::mul(vxx, vxx, v);
  f32o::sub(m_root_check, vxx, u); /* vx^2-u */
  f32o::add(p_root_check, vxx, u); /* vx^2+u */
  has_m_root = f32p::is_zero(m_root_check);
  has_p_root = f32p::is_zero(p_root_check);
  f32o::mul(x_sqrtm1, h.X, f32cn::sqrtm1_v); /* x*sqrt(-1) */
  f32o::cmov(h.X, x_sqrtm1, 1 - has_m_root);

  f32o::neg(negx, h.X);
  f32o::cmov(h.X, negx, f32p::is_negative(h.X) ^ (s[31] >> 7));
  f32o::mul(h.T, h.X, h.Y);

  return (has_m_root | has_p_root) - 1;
}
} // namespace sxt::c32t
