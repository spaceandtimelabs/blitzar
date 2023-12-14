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
#include "sxt/curve_g1/operation/add.h"

#include "sxt/curve_g1/operation/cmov.h"
#include "sxt/curve_g1/property/identity.h"
#include "sxt/curve_g1/type/element_affine.h"

namespace sxt::cg1o {
//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void add(cg1t::element_p2& h, const cg1t::element_p2& p, const cg1t::element_affine& q) noexcept {
  f12t::element t0, t1, t2, t3, t4;
  f12t::element x3, y3, z3;

  f12o::mul(t0, p.X, q.X);
  f12o::mul(t1, p.Y, q.Y);
  fmtgo::add(t3, q.X, q.Y);
  fmtgo::add(t4, p.X, p.Y);
  f12o::mul(t3, t3, t4);
  fmtgo::add(t4, t0, t1);
  f12o::sub(t3, t3, t4);
  f12o::mul(t4, q.Y, p.Z);
  fmtgo::add(t4, t4, p.Y);
  f12o::mul(y3, q.X, p.Z);
  fmtgo::add(y3, y3, p.X);
  fmtgo::add(x3, t0, t0);
  fmtgo::add(t0, x3, t0);
  mul_by_3b(t2, p.Z);
  fmtgo::add(z3, t1, t2);
  f12o::sub(t1, t1, t2);
  mul_by_3b(y3, y3);
  f12o::mul(x3, t4, y3);
  f12o::mul(t2, t3, t1);
  f12o::sub(x3, t2, x3);
  f12o::mul(y3, y3, t0);
  f12o::mul(t1, t1, z3);
  fmtgo::add(y3, t1, y3);
  f12o::mul(t0, t0, t3);
  f12o::mul(z3, z3, t4);
  fmtgo::add(z3, z3, t0);

  h.X = x3;
  h.Y = y3;
  h.Z = z3;

  cmov(h, p, cg1p::is_identity(q));
}
} // namespace sxt::cg1o
