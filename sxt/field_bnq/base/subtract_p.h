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
#pragma once

#include "sxt/base/field/arithmetic_utility.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/num/cmov.h"
#include "sxt/field_bnq/base/constants.h"

namespace sxt::fbnqb {
//--------------------------------------------------------------------------------------------------
// subtract_p
//--------------------------------------------------------------------------------------------------
/*
 Compute ret = a - p_v, where p_v is the modulus.
 */
CUDA_CALLABLE inline void subtract_p(uint64_t ret[4], const uint64_t a[4]) noexcept {
  uint64_t borrow{0};

  basfld::sbb(ret[0], borrow, a[0], p_v[0]);
  basfld::sbb(ret[1], borrow, a[1], p_v[1]);
  basfld::sbb(ret[2], borrow, a[2], p_v[2]);
  basfld::sbb(ret[3], borrow, a[3], p_v[3]);

  // If underflow occurred on the final limb, borrow = 0xfff...fff, otherwise
  // borrow = 0x000...000. Thus, we use it as a mask!
  uint64_t mask{0x0};
  basn::cmov(mask, borrow - 1, borrow == 0x0);

  ret[0] = (a[0] & borrow) | (ret[0] & mask);
  ret[1] = (a[1] & borrow) | (ret[1] & mask);
  ret[2] = (a[2] & borrow) | (ret[2] & mask);
  ret[3] = (a[3] & borrow) | (ret[3] & mask);
}
} // namespace sxt::fbnqb
