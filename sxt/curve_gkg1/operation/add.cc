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
#include "sxt/curve_gkg1/operation/add.h"

#include "sxt/curve_gkg1/operation/cmov.h"
#include "sxt/curve_gkg1/property/identity.h"
#include "sxt/curve_gkg1/type/element_affine.h"

namespace sxt::ck1o {
//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void add(ck1t::element_p2& h, const ck1t::element_p2& p, const ck1t::element_affine& q) noexcept {
  fgkt::element t0, t1, t2, t3, t4;
  fgkt::element x3, y3, z3;

  fgko::mul(t0, p.X, q.X);
  fgko::mul(t1, p.Y, q.Y);
  fgko::add(t3, q.X, q.Y);
  fgko::add(t4, p.X, p.Y);
  fgko::mul(t3, t3, t4);
  fgko::add(t4, t0, t1);
  fgko::sub(t3, t3, t4);
  fgko::mul(t4, q.Y, p.Z);
  fgko::add(t4, t4, p.Y);
  fgko::mul(y3, q.X, p.Z);
  fgko::add(y3, y3, p.X);
  fgko::add(x3, t0, t0);
  fgko::add(t0, x3, t0);
  mul_by_3b(t2, p.Z);
  fgko::add(z3, t1, t2);
  fgko::sub(t1, t1, t2);
  mul_by_3b(y3, y3);
  fgko::mul(x3, t4, y3);
  fgko::mul(t2, t3, t1);
  fgko::sub(x3, t2, x3);
  fgko::mul(y3, y3, t0);
  fgko::mul(t1, t1, z3);
  fgko::add(y3, t1, y3);
  fgko::mul(t0, t0, t3);
  fgko::mul(z3, z3, t4);
  fgko::add(z3, z3, t0);

  h.X = x3;
  h.Y = y3;
  h.Z = z3;

  cmov(h, p, ck1p::is_identity(q));
}
} // namespace sxt::ck1o
