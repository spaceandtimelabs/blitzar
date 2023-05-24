/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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

#include "sxt/curve21/operation/add.h"

#include "sxt/field51/operation/add.h"
#include "sxt/field51/operation/mul.h"
#include "sxt/field51/operation/sub.h"
#include "sxt/field51/type/element.h"

namespace sxt::c21o {
//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
/*
 r = p + q
 */
CUDA_CALLABLE
void add(c21t::element_p1p1& r, const c21t::element_p3& p, const c21t::element_cached& q) noexcept {
  f51t::element t0;

  f51o::add(r.X, p.Y, p.X);
  f51o::sub(r.Y, p.Y, p.X);
  f51o::mul(r.Z, r.X, q.YplusX);
  f51o::mul(r.Y, r.Y, q.YminusX);
  f51o::mul(r.T, q.T2d, p.T);
  f51o::mul(r.X, p.Z, q.Z);
  f51o::add(t0, r.X, r.X);
  f51o::sub(r.X, r.Z, r.Y);
  f51o::add(r.Y, r.Z, r.Y);
  f51o::add(r.Z, t0, r.T);
  f51o::sub(r.T, t0, r.T);
}
} // namespace sxt::c21o
