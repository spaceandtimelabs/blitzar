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
 * Adopted from libsodium
 *
 * Copyright (c) 2013-2023
 * Frank Denis <j at pureftpd dot org>
 *
 * See third_party/license/libsodium.LICENSE
 */

#include "sxt/curve32/operation/add.h"

#include "sxt/field32/operation/add.h"
#include "sxt/field32/operation/mul.h"
#include "sxt/field32/operation/sub.h"
#include "sxt/field32/type/element.h"

namespace sxt::c32o {
//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
/*
 r = p + q
 */
CUDA_CALLABLE
void add(c32t::element_p1p1& r, const c32t::element_p3& p, const c32t::element_cached& q) noexcept {
  f32t::element t0;

  f32o::add(r.X, p.Y, p.X);
  f32o::sub(r.Y, p.Y, p.X);
  f32o::mul(r.Z, r.X, q.YplusX);
  f32o::mul(r.Y, r.Y, q.YminusX);
  f32o::mul(r.T, q.T2d, p.T);
  f32o::mul(r.X, p.Z, q.Z);
  f32o::add(t0, r.X, r.X);
  f32o::sub(r.X, r.Z, r.Y);
  f32o::add(r.Y, r.Z, r.Y);
  f32o::add(r.Z, t0, r.T);
  f32o::sub(r.T, t0, r.T);
}
} // namespace sxt::c32o
