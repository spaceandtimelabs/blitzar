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
#include "sxt/curve_bng1_32/operation/double.h"

#include "sxt/curve_bng1_32/operation/cmov.h"
#include "sxt/curve_bng1_32/operation/mul_by_3b.h"
#include "sxt/curve_bng1_32/property/identity.h"
#include "sxt/curve_bng1_32/type/element_p2.h"
#include "sxt/field32/operation/add.h"
#include "sxt/field32/operation/mul.h"
#include "sxt/field32/operation/square.h"
#include "sxt/field32/operation/sub.h"
#include "sxt/field32/type/element.h"

namespace sxt::cn3o {
//--------------------------------------------------------------------------------------------------
// double_element
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void double_element(cn3t::element_p2& h, const cn3t::element_p2& p) noexcept {
  f32t::element t0, t1, t2;
  f32t::element x3, y3, z3;

  f32o::square(t0, p.Y);
  f32o::add(z3, t0, t0);
  f32o::add(z3, z3, z3);
  f32o::add(z3, z3, z3);
  f32o::mul(t1, p.Y, p.Z);
  f32o::square(t2, p.Z);
  mul_by_3b(t2, t2);
  f32o::mul(x3, t2, z3);
  f32o::add(y3, t0, t2);
  f32o::mul(z3, t1, z3);
  f32o::add(t1, t2, t2);
  f32o::add(t2, t1, t2);
  f32o::sub(t0, t0, t2);
  f32o::mul(y3, t0, y3);
  f32o::add(y3, x3, y3);
  f32o::mul(t1, p.X, p.Y);
  f32o::mul(x3, t0, t1);
  f32o::add(x3, x3, x3);

  h.X = x3;
  h.Y = y3;
  h.Z = z3;

  cmov(h, cn3t::element_p2::identity(), cn3p::is_identity(p));
}
} // namespace sxt::cn3o