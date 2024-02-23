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

#include "sxt/curve32/type/double_impl.h"

#include "sxt/curve32/type/element_p1p1.h"
#include "sxt/curve32/type/element_p2.h"
#include "sxt/field32/operation/add.h"
#include "sxt/field32/operation/sq.h"
#include "sxt/field32/operation/sub.h"
#include "sxt/field32/type/element.h"

namespace sxt::c32t {
//--------------------------------------------------------------------------------------------------
// double_element_impl
//--------------------------------------------------------------------------------------------------
/*
 r = 2 * p
*/
CUDA_CALLABLE
void double_element_impl(c32t::element_p1p1& r, const c32t::element_p2& p) noexcept {
  f32t::element t0;

  f32o::sq(r.X, p.X);
  f32o::sq(r.Z, p.Y);
  f32o::sq2(r.T, p.Z);
  f32o::add(r.Y, p.X, p.Y);
  f32o::sq(t0, r.Y);
  f32o::add(r.Y, r.Z, r.X);
  f32o::sub(r.Z, r.Z, r.X);
  f32o::sub(r.X, t0, r.Y);
  f32o::sub(r.T, r.T, r.Z);
}
} // namespace sxt::c32t
