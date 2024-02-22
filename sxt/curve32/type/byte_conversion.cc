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
#include "sxt/field51/base/byte_conversion.h"
#include "sxt/field51/constant/d.h"
#include "sxt/field51/constant/one.h"
#include "sxt/field51/constant/sqrtm1.h"
#include "sxt/field51/operation/add.h"
#include "sxt/field51/operation/cmov.h"
#include "sxt/field51/operation/invert.h"
#include "sxt/field51/operation/mul.h"
#include "sxt/field51/operation/neg.h"
#include "sxt/field51/operation/pow22523.h"
#include "sxt/field51/operation/sq.h"
#include "sxt/field51/operation/sub.h"
#include "sxt/field51/property/sign.h"
#include "sxt/field51/property/zero.h"
#include "sxt/field51/type/element.h"

namespace sxt::c32t {
//--------------------------------------------------------------------------------------------------
// to_bytes
//--------------------------------------------------------------------------------------------------
void to_bytes(uint8_t s[32], const element_p3& h) noexcept {
  f51t::element recip;
  f51t::element x;
  f51t::element y;

  f51o::invert(recip, h.Z);
  f51o::mul(x, h.X, recip);
  f51o::mul(y, h.Y, recip);
  f51b::to_bytes(s, y.data());
  s[31] ^= f51p::is_negative(x) << 7;
}

//--------------------------------------------------------------------------------------------------
// from_bytes
//--------------------------------------------------------------------------------------------------
int from_bytes(element_p3& h, uint8_t s[32]) noexcept {
  f51t::element u;
  f51t::element v;
  f51t::element vxx;
  f51t::element m_root_check, p_root_check;
  f51t::element negx;
  f51t::element x_sqrtm1;
  int has_m_root, has_p_root;

  f51b::from_bytes(h.Y.data(), s);
  h.Z = f51cn::one_v;
  f51o::sq(u, h.Y);
  f51o::mul(v, u, f51cn::d_v);
  f51o::sub(u, u, h.Z); /* u = y^2-1 */
  f51o::add(v, v, h.Z); /* v = dy^2+1 */

  f51o::mul(h.X, u, v);
  f51o::pow22523(h.X, h.X);
  f51o::mul(h.X, u, h.X); /* u((uv)^((q-5)/8)) */

  f51o::sq(vxx, h.X);
  f51o::mul(vxx, vxx, v);
  f51o::sub(m_root_check, vxx, u); /* vx^2-u */
  f51o::add(p_root_check, vxx, u); /* vx^2+u */
  has_m_root = f51p::is_zero(m_root_check);
  has_p_root = f51p::is_zero(p_root_check);
  f51o::mul(x_sqrtm1, h.X, f51cn::sqrtm1_v); /* x*sqrt(-1) */
  f51o::cmov(h.X, x_sqrtm1, 1 - has_m_root);

  f51o::neg(negx, h.X);
  f51o::cmov(h.X, negx, f51p::is_negative(h.X) ^ (s[31] >> 7));
  f51o::mul(h.T, h.X, h.Y);

  return (has_m_root | has_p_root) - 1;
}
} // namespace sxt::c32t
