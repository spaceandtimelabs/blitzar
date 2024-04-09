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
#include "sxt/curve_bng1_32/operation/add.h"

#include "sxt/curve_bng1_32/operation/cmov.h"
#include "sxt/curve_bng1_32/property/identity.h"
#include "sxt/curve_bng1_32/type/element_affine.h"

namespace sxt::cn3o {
//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void add(cn3t::element_p2& h, const cn3t::element_p2& p, const cn3t::element_affine& q) noexcept {
  f32t::element t0, t1, t2, t3, t4;
  f32t::element x3, y3, z3;

  f32o::mul(t0, p.X, q.X);
  f32o::mul(t1, p.Y, q.Y);
  f32o::add(t3, q.X, q.Y);
  f32o::add(t4, p.X, p.Y);
  f32o::mul(t3, t3, t4);
  f32o::add(t4, t0, t1);
  f32o::sub(t3, t3, t4);
  f32o::mul(t4, q.Y, p.Z);
  f32o::add(t4, t4, p.Y);
  f32o::mul(y3, q.X, p.Z);
  f32o::add(y3, y3, p.X);
  f32o::add(x3, t0, t0);
  f32o::add(t0, x3, t0);
  mul_by_3b(t2, p.Z);
  f32o::add(z3, t1, t2);
  f32o::sub(t1, t1, t2);
  mul_by_3b(y3, y3);
  f32o::mul(x3, t4, y3);
  f32o::mul(t2, t3, t1);
  f32o::sub(x3, t2, x3);
  f32o::mul(y3, y3, t0);
  f32o::mul(t1, t1, z3);
  f32o::add(y3, t1, y3);
  f32o::mul(t0, t0, t3);
  f32o::mul(z3, z3, t4);
  f32o::add(z3, z3, t0);

  h.X = x3;
  h.Y = y3;
  h.Z = z3;

  cmov(h, p, cn3p::is_identity(q));
}
} // namespace sxt::cn3o
