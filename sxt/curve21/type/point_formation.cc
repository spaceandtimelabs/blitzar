/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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
 * Copyright (c) 2013-2022
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */
#include "sxt/curve21/type/point_formation.h"

#include <cstring>

#include "sxt/curve21/base/elligate.h"
#include "sxt/curve21/base/mont_ed_conversion.h"
#include "sxt/curve21/type/cofactor_utility.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/field51/base/byte_conversion.h"
#include "sxt/field51/constant/one.h"
#include "sxt/field51/operation/cmov.h"
#include "sxt/field51/operation/mul.h"
#include "sxt/field51/operation/neg.h"
#include "sxt/field51/property/sign.h"
#include "sxt/field51/type/element.h"

namespace sxt::c21t {
//--------------------------------------------------------------------------------------------------
// form_point
//--------------------------------------------------------------------------------------------------
void form_point(element_p3& p, const uint8_t r[32]) noexcept {
  element_p3 p3;
  f51t::element x, y, negxed;
  f51t::element r_fe;
  int notsquare;
  unsigned char x_sign;

  uint8_t s[32];
  std::memcpy(s, r, 32);
  x_sign = s[31] >> 7;
  s[31] &= 0x7f;
  f51b::from_bytes(r_fe.data(), s);

  c21b::apply_elligator(x, y, &notsquare, r_fe);

  c21b::mont_to_ed(p3.X, p3.Y, x, y);
  f51o::neg(negxed, p3.X);
  f51o::cmov(p3.X, negxed, f51p::is_negative(p3.X) ^ x_sign);

  p3.Z = f51cn::one_v;
  f51o::mul(p3.T, p3.X, p3.Y);
  clear_cofactor(p3);

  p = p3;
}
} // namespace sxt::c21t
