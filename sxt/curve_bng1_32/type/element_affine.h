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
#pragma once

#include "sxt/field32/constant/one.h"
#include "sxt/field32/constant/zero.h"
#include "sxt/field32/type/element.h"

namespace sxt::cn3t {
//--------------------------------------------------------------------------------------------------
// element_affine
//--------------------------------------------------------------------------------------------------
/**
 * This is an element of G1 represented in the affine coordinate space.
 * It is ideal to keep elements in this representation to reduce memory usage and
 * improve performance through the use of mixed curve model arithmetic.
 * Values of `G1Affine` are guaranteed to be in the q-order subgroup.
 */
struct element_affine {
  f32t::element X;
  f32t::element Y;
  uint8_t infinity;

  static constexpr element_affine identity() noexcept {
    return element_affine{f32cn::zero_v, f32cn::one_v, true};
  }

  bool operator==(const element_affine& rhs) const noexcept = default;
};
} // namespace sxt::cn3t
