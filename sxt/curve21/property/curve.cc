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
 * Copyright (c) 2013-2023
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */

#include "sxt/curve21/property/curve.h"

#include "sxt/curve21/type/element_p3.h"
#include "sxt/field51/constant/d.h"
#include "sxt/field51/operation/add.h"
#include "sxt/field51/operation/mul.h"
#include "sxt/field51/operation/sq.h"
#include "sxt/field51/operation/sub.h"
#include "sxt/field51/property/zero.h"
#include "sxt/field51/type/element.h"

namespace sxt::c21p {
//--------------------------------------------------------------------------------------------------
// is_on_curve
//--------------------------------------------------------------------------------------------------
bool is_on_curve(const c21t::element_p3& p) noexcept {
  f51t::element x2;
  f51t::element y2;
  f51t::element z2;
  f51t::element z4;
  f51t::element t0;
  f51t::element t1;

  f51o::sq(x2, p.X);
  f51o::sq(y2, p.Y);
  f51o::sq(z2, p.Z);
  f51o::sub(t0, y2, x2);
  f51o::mul(t0, t0, z2);

  f51o::mul(t1, x2, y2);
  f51o::mul(t1, t1, f51t::element{f51cn::d_v});
  f51o::sq(z4, z2);
  f51o::add(t1, t1, z4);
  f51o::sub(t0, t0, t1);

  return f51p::is_zero(t0);
}
} // namespace sxt::c21p
