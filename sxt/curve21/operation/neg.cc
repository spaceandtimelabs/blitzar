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

#include "sxt/curve21/operation/neg.h"

#include "sxt/curve21/type/element_p3.h"
#include "sxt/field51/operation/cmov.h"
#include "sxt/field51/operation/neg.h"

namespace sxt::c21o {
//--------------------------------------------------------------------------------------------------
// neg
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void neg(c21t::element_p3& r, const c21t::element_p3& p) noexcept {
  f51o::neg(r.X, p.X);
  r.Y = p.Y;
  r.Z = p.Z;
  f51o::neg(r.T, p.T);
}

//--------------------------------------------------------------------------------------------------
// cneg
//--------------------------------------------------------------------------------------------------
/* r = -r if b = 1 else r */
CUDA_CALLABLE
void cneg(c21t::element_p3& r, unsigned int b) noexcept {
  f51t::element t;
  f51o::neg(t, r.X);
  f51o::cmov(r.X, t, b);
  f51o::neg(t, r.T);
  f51o::cmov(r.T, t, b);
}
} // namespace sxt::c21o
