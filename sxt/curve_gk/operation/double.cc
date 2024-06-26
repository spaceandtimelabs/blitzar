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
#include "sxt/curve_gk/operation/double.h"

#include "sxt/curve_gk/operation/cmov.h"
#include "sxt/curve_gk/operation/mul_by_3b.h"
#include "sxt/curve_gk/property/identity.h"
#include "sxt/curve_gk/type/element_p2.h"
#include "sxt/fieldgk/operation/add.h"
#include "sxt/fieldgk/operation/mul.h"
#include "sxt/fieldgk/operation/square.h"
#include "sxt/fieldgk/operation/sub.h"
#include "sxt/fieldgk/type/element.h"

namespace sxt::cgko {
//--------------------------------------------------------------------------------------------------
// double_element
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void double_element(cgkt::element_p2& h, const cgkt::element_p2& p) noexcept {
  fgkt::element t0, t1, t2;
  fgkt::element x3, y3, z3;

  fgko::square(t0, p.Y);
  fgko::add(z3, t0, t0);
  fgko::add(z3, z3, z3);
  fgko::add(z3, z3, z3);
  fgko::mul(t1, p.Y, p.Z);
  fgko::square(t2, p.Z);
  mul_by_3b(t2, t2);
  fgko::mul(x3, t2, z3);
  fgko::add(y3, t0, t2);
  fgko::mul(z3, t1, z3);
  fgko::add(t1, t2, t2);
  fgko::add(t2, t1, t2);
  fgko::sub(t0, t0, t2);
  fgko::mul(y3, t0, y3);
  fgko::add(y3, x3, y3);
  fgko::mul(t1, p.X, p.Y);
  fgko::mul(x3, t0, t1);
  fgko::add(x3, x3, x3);

  h.X = x3;
  h.Y = y3;
  h.Z = z3;

  cmov(h, cgkt::element_p2::identity(), cgkp::is_identity(p));
}
} // namespace sxt::cgko
