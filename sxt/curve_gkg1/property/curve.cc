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
#include "sxt/curve_gkg1/property/curve.h"

#include "sxt/curve_gkg1/constant/b.h"
#include "sxt/curve_gkg1/type/element_affine.h"
#include "sxt/curve_gkg1/type/element_p2.h"
#include "sxt/field25/operation/add.h"
#include "sxt/field25/operation/mul.h"
#include "sxt/field25/operation/square.h"
#include "sxt/field25/operation/sub.h"
#include "sxt/field25/property/zero.h"
#include "sxt/field25/type/element.h"

namespace sxt::ck1p {
//--------------------------------------------------------------------------------------------------
// is_on_curve
//--------------------------------------------------------------------------------------------------
/**
 * Returns true if the element is on the curve: y^2 - x^3 = b_v
 */
bool is_on_curve(const ck1t::element_affine& p) noexcept {
  f25t::element y2;
  f25o::square(y2, p.Y);

  f25t::element x2;
  f25t::element x3;
  f25o::square(x2, p.X);
  f25o::mul(x3, x2, p.X);

  f25t::element y2_x3;
  f25o::sub(y2_x3, y2, x3);

  return (y2_x3 == ck1cn::b_v) || p.infinity;
}

//--------------------------------------------------------------------------------------------------
// is_on_curve
//--------------------------------------------------------------------------------------------------
/**
 * Returns true if the element is on the curve: (y^2 * z) = x^3 + (b_v * z^3)
 */
bool is_on_curve(const ck1t::element_p2& p) noexcept {
  f25t::element y2;
  f25t::element y2_z;
  f25o::square(y2, p.Y);
  f25o::mul(y2_z, y2, p.Z);

  f25t::element x2;
  f25t::element x3;
  f25o::square(x2, p.X);
  f25o::mul(x3, x2, p.X);

  f25t::element z2;
  f25t::element z3;
  f25t::element b_z3;
  f25o::square(z2, p.Z);
  f25o::mul(z3, z2, p.Z);
  f25o::mul(b_z3, f25t::element{ck1cn::b_v}, z3);

  f25t::element x3_b_z3;
  f25o::add(x3_b_z3, x3, b_z3);

  return (y2_z == x3_b_z3) || f25p::is_zero(p.Z);
}
} // namespace sxt::ck1p
