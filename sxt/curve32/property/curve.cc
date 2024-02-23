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

#include "sxt/curve32/property/curve.h"

#include "sxt/curve32/type/element_p3.h"
#include "sxt/field32/constant/d.h"
#include "sxt/field32/operation/add.h"
#include "sxt/field32/operation/mul.h"
#include "sxt/field32/operation/sq.h"
#include "sxt/field32/operation/sub.h"
#include "sxt/field32/property/zero.h"
#include "sxt/field32/type/element.h"

namespace sxt::c32p {
//--------------------------------------------------------------------------------------------------
// is_on_curve
//--------------------------------------------------------------------------------------------------
bool is_on_curve(const c32t::element_p3& p) noexcept {
  f32t::element x2;
  f32t::element y2;
  f32t::element z2;
  f32t::element z4;
  f32t::element t0;
  f32t::element t1;

  f32o::sq(x2, p.X);
  f32o::sq(y2, p.Y);
  f32o::sq(z2, p.Z);
  f32o::sub(t0, y2, x2);
  f32o::mul(t0, t0, z2);

  f32o::mul(t1, x2, y2);
  f32o::mul(t1, t1, f32t::element{f32cn::d_v});
  f32o::sq(z4, z2);
  f32o::add(t1, t1, z4);
  f32o::sub(t0, t0, t1);

  return f32p::is_zero(t0);
}
} // namespace sxt::c32p
