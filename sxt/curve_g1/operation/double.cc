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
/*
 * Adopted from zkcrypto/bls12_381
 *
 * Copyright (c) 2021
 * Sean Bowe <ewillbefull@gmail.com>
 * Jack Grigg <thestr4d@gmail.com>
 *
 * See third_party/license/zkcrypto.LICENSE
 */
#include "sxt/curve_g1/operation/double.h"

#include "sxt/curve_g1/operation/cmov.h"
#include "sxt/curve_g1/operation/mul_by_3b.h"
#include "sxt/curve_g1/property/identity.h"
#include "sxt/curve_g1/type/element_p2.h"
#include "sxt/field12/operation/mul.h"
#include "sxt/field12/operation/square.h"
#include "sxt/field12/operation/sub.h"
#include "sxt/field12/type/element.h"
#include "sxt/field_mtg/operation/add.h"

namespace sxt::cg1o {
//--------------------------------------------------------------------------------------------------
// double_element
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void double_element(cg1t::element_p2& h, const cg1t::element_p2& p) noexcept {
  f12t::element t0, t1, t2;
  f12t::element x3, y3, z3;

  f12o::square(t0, p.Y);
  fmtgo::add(z3, t0, t0);
  fmtgo::add(z3, z3, z3);
  fmtgo::add(z3, z3, z3);
  f12o::mul(t1, p.Y, p.Z);
  f12o::square(t2, p.Z);
  mul_by_3b(t2, t2);
  f12o::mul(x3, t2, z3);
  fmtgo::add(y3, t0, t2);
  f12o::mul(z3, t1, z3);
  fmtgo::add(t1, t2, t2);
  fmtgo::add(t2, t1, t2);
  f12o::sub(t0, t0, t2);
  f12o::mul(y3, t0, y3);
  fmtgo::add(y3, x3, y3);
  f12o::mul(t1, p.X, p.Y);
  f12o::mul(x3, t0, t1);
  fmtgo::add(x3, x3, x3);

  h.X = x3;
  h.Y = y3;
  h.Z = z3;

  cmov(h, cg1t::element_p2::identity(), cg1p::is_identity(p));
}
} // namespace sxt::cg1o
