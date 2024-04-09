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
 * Adopted from zkcrypto/bls12_381
 *
 * Copyright (c) 2021
 * Sean Bowe <ewillbefull@gmail.com>
 * Jack Grigg <thestr4d@gmail.com>
 *
 * See third_party/license/zkcrypto.LICENSE
 */
#include "sxt/curve_bng1_32/property/curve.h"

#include "sxt/curve_bng1_32/constant/b.h"
#include "sxt/curve_bng1_32/type/element_affine.h"
#include "sxt/curve_bng1_32/type/element_p2.h"
#include "sxt/field32/operation/add.h"
#include "sxt/field32/operation/mul.h"
#include "sxt/field32/operation/square.h"
#include "sxt/field32/operation/sub.h"
#include "sxt/field32/property/zero.h"
#include "sxt/field32/type/element.h"

namespace sxt::cn3p {
//--------------------------------------------------------------------------------------------------
// is_on_curve
//--------------------------------------------------------------------------------------------------
/**
 * Returns true if the element is on the curve: y^2 - x^3 = b_v
 */
bool is_on_curve(const cn3t::element_affine& p) noexcept {
  f32t::element y2;
  f32o::square(y2, p.Y);

  f32t::element x2;
  f32t::element x3;
  f32o::square(x2, p.X);
  f32o::mul(x3, x2, p.X);

  f32t::element y2_x3;
  f32o::sub(y2_x3, y2, x3);

  return (y2_x3 == cn3cn::b_v) || p.infinity;
}

//--------------------------------------------------------------------------------------------------
// is_on_curve
//--------------------------------------------------------------------------------------------------
/**
 * Returns true if the element is on the curve: (y^2 * z) = x^3 + (b_v * z^3)
 */
bool is_on_curve(const cn3t::element_p2& p) noexcept {
  f32t::element y2;
  f32t::element y2_z;
  f32o::square(y2, p.Y);
  f32o::mul(y2_z, y2, p.Z);

  f32t::element x2;
  f32t::element x3;
  f32o::square(x2, p.X);
  f32o::mul(x3, x2, p.X);

  f32t::element z2;
  f32t::element z3;
  f32t::element b_z3;
  f32o::square(z2, p.Z);
  f32o::mul(z3, z2, p.Z);
  f32o::mul(b_z3, f32t::element{cn3cn::b_v}, z3);

  f32t::element x3_b_z3;
  f32o::add(x3_b_z3, x3, b_z3);

  return (y2_z == x3_b_z3) || f32p::is_zero(p.Z);
}
} // namespace sxt::cn3p
