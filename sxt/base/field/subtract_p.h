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
/*
 * Adopted from zkcrypto/bls12_381
 *
 * Copyright (c) 2021
 * Sean Bowe <ewillbefull@gmail.com>
 * Jack Grigg <thestr4d@gmail.com>
 *
 * See third_party/license/zkcrypto.LICENSE
 */
#pragma once

#include "sxt/base/field/arithmetic_utility.h"
#include "sxt/base/macro/cuda_callable.h"

namespace sxt::basf {
//--------------------------------------------------------------------------------------------------
// subtract_p
//--------------------------------------------------------------------------------------------------
/*
 Compute ret = a - p, where p is the modulus.
 */
template <const size_t limbs>
CUDA_CALLABLE inline void subtract_p(uint64_t* ret, const uint64_t* const a,
                                     const uint64_t* const p) noexcept {
  uint64_t borrow{0};

  for (size_t limb{0}; limb < limbs; ++limb) {
    sbb(ret[limb], borrow, a[limb], p[limb]);
  }

  // If underflow occurred on the final limb, borrow = 0xfff...fff, otherwise
  // borrow = 0x000...000. Thus, we use it as a mask!
  uint64_t mask{borrow == 0x0 ? (borrow - 1) : 0x0};

  for (size_t limb{0}; limb < limbs; ++limb) {
    ret[limb] = (a[limb] & borrow) | (ret[limb] & mask);
  }
}
} // namespace sxt::basf
