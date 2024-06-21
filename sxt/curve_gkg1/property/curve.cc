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
#include "sxt/fieldgk/operation/add.h"
#include "sxt/fieldgk/operation/mul.h"
#include "sxt/fieldgk/operation/square.h"
#include "sxt/fieldgk/operation/sub.h"
#include "sxt/fieldgk/property/zero.h"
#include "sxt/fieldgk/type/element.h"

namespace sxt::ck1p {
//--------------------------------------------------------------------------------------------------
// is_on_curve
//--------------------------------------------------------------------------------------------------
/**
 * Returns true if the element is on the curve: y^2 - x^3 = b_v
 */
bool is_on_curve(const ck1t::element_affine& p) noexcept {
  fgkt::element y2;
  fgko::square(y2, p.Y);

  fgkt::element x2;
  fgkt::element x3;
  fgko::square(x2, p.X);
  fgko::mul(x3, x2, p.X);

  fgkt::element y2_x3;
  fgko::sub(y2_x3, y2, x3);

  return (y2_x3 == ck1cn::b_v) || p.infinity;
}

//--------------------------------------------------------------------------------------------------
// is_on_curve
//--------------------------------------------------------------------------------------------------
/**
 * Returns true if the element is on the curve: (y^2 * z) = x^3 + (b_v * z^3)
 */
bool is_on_curve(const ck1t::element_p2& p) noexcept {
  fgkt::element y2;
  fgkt::element y2_z;
  fgko::square(y2, p.Y);
  fgko::mul(y2_z, y2, p.Z);

  fgkt::element x2;
  fgkt::element x3;
  fgko::square(x2, p.X);
  fgko::mul(x3, x2, p.X);

  fgkt::element z2;
  fgkt::element z3;
  fgkt::element b_z3;
  fgko::square(z2, p.Z);
  fgko::mul(z3, z2, p.Z);
  fgko::mul(b_z3, fgkt::element{ck1cn::b_v}, z3);

  fgkt::element x3_b_z3;
  fgko::add(x3_b_z3, x3, b_z3);

  return (y2_z == x3_b_z3) || fgkp::is_zero(p.Z);
}
} // namespace sxt::ck1p
