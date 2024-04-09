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
 * Adopted from zcash/librustzcash
 *
 * Copyright (c) 2017
 * Zcash Company
 *
 * See third_party/license/zcash.LICENSE
 */
#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/num/cmov.h"
#include "sxt/curve_bng1_32/type/element_affine.h"
#include "sxt/curve_bng1_32/type/element_p2.h"
#include "sxt/field32/operation/cmov.h"
#include "sxt/field32/operation/invert.h"
#include "sxt/field32/operation/mul.h"
#include "sxt/field32/type/element.h"

namespace sxt::cn3t {
//--------------------------------------------------------------------------------------------------
// to_element_affine
//--------------------------------------------------------------------------------------------------
/**
 * Converts projective to affine element.
 */
CUDA_CALLABLE
inline void to_element_affine(element_affine& a, const element_p2& p) noexcept {
  f32t::element z_inv;
  const bool is_zero{f32o::invert(z_inv, p.Z)};
  f32o::cmov(z_inv, f32cn::zero_v, is_zero);

  f32t::element x;
  f32t::element y;
  f32o::mul(x, p.X, z_inv);
  f32o::mul(y, p.Y, z_inv);

  a.X = x;
  a.Y = y;
  a.infinity = false;

  f32o::cmov(a.X, element_affine::identity().X, is_zero);
  f32o::cmov(a.Y, element_affine::identity().Y, is_zero);
  basn::cmov(a.infinity, element_affine::identity().infinity, is_zero);
}

//--------------------------------------------------------------------------------------------------
// to_element_p2
//--------------------------------------------------------------------------------------------------
/**
 * Converts affine to projective element.
 */
CUDA_CALLABLE
inline void to_element_p2(element_p2& p, const element_affine& a) noexcept {
  p.X = a.X;
  p.Y = a.Y;
  p.Z = f32cn::one_v;
  f32o::cmov(p.Z, f32cn::zero_v, a.infinity);
}

//--------------------------------------------------------------------------------------------------
// batch_to_element_p2
//--------------------------------------------------------------------------------------------------
/**
 * Batch converts projective to affine element.
 */
CUDA_CALLABLE
inline void batch_to_element_affine(basct::span<cn3t::element_affine> a,
                                    basct::cspan<cn3t::element_p2> p) noexcept {
  SXT_DEBUG_ASSERT(a.size() == p.size());
  for (size_t i = 0; i < p.size(); ++i) {
    to_element_affine(a[i], p[i]);
  }
}

//--------------------------------------------------------------------------------------------------
// batch_to_element_p2
//--------------------------------------------------------------------------------------------------
/**
 * Batch converts affine to projective element.
 */
CUDA_CALLABLE
inline void batch_to_element_p2(basct::span<cn3t::element_p2> p,
                                basct::cspan<cn3t::element_affine> a) noexcept {
  SXT_DEBUG_ASSERT(a.size() == p.size());
  for (size_t i = 0; i < a.size(); ++i) {
    to_element_p2(p[i], a[i]);
  }
}
} // namespace sxt::cn3t
