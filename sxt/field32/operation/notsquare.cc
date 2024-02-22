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
#include "sxt/field32/operation/notsquare.h"

#include "sxt/field32/base/byte_conversion.h"
#include "sxt/field32/operation/mul.h"
#include "sxt/field32/operation/sq.h"
#include "sxt/field32/operation/sqmul.h"
#include "sxt/field32/type/element.h"

namespace sxt::f32o {
//--------------------------------------------------------------------------------------------------
// notsquare
//--------------------------------------------------------------------------------------------------
int notsquare(const f32t::element& x) noexcept {
  f32t::element _10, _11, _1100, _1111, _11110000, _11111111;
  f32t::element t, u, v;
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

  f32b::to_bytes(s, t.data());

  return s[1] & 1;
}
} // namespace sxt::f32o
