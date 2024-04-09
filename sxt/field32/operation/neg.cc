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
#include "sxt/field32/operation/neg.h"

#include "sxt/base/field/arithmetic_utility.h"
#include "sxt/field32/base/constants.h"
#include "sxt/field32/type/element.h"

namespace sxt::f32o {
//--------------------------------------------------------------------------------------------------
// neg
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void neg(f32t::element& h, const f32t::element& f) noexcept {
  uint32_t d[8] = {};
  uint32_t borrow{0};

  basfld::sbb(d[0], borrow, f32b::p_v[0], f[0]);
  basfld::sbb(d[1], borrow, f32b::p_v[1], f[1]);
  basfld::sbb(d[2], borrow, f32b::p_v[2], f[2]);
  basfld::sbb(d[3], borrow, f32b::p_v[3], f[3]);
  basfld::sbb(d[4], borrow, f32b::p_v[4], f[4]);
  basfld::sbb(d[5], borrow, f32b::p_v[5], f[5]);
  basfld::sbb(d[6], borrow, f32b::p_v[6], f[6]);
  basfld::sbb(d[7], borrow, f32b::p_v[7], f[7]);

  // Let's use a mask if f was zero, which would mean
  // the result of the subtraction is p.
  uint32_t mask =
      uint32_t{((f[0] | f[1] | f[2] | f[3] | f[4] | f[5] | f[6] | f[7]) == 0)} - uint32_t{1};

  for (size_t i = 0; i < f.num_limbs_v; ++i) {
    h[i] = d[i] & mask;
  }
}
} // namespace sxt::f32o
