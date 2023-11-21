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

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/type/int.h"
#include "sxt/base/type/narrow_cast.h"

namespace sxt::basf {
//--------------------------------------------------------------------------------------------------
// mac
//--------------------------------------------------------------------------------------------------
/*
 Compute a + (b * c) + carry, returning the result and the new carry over.
 */
CUDA_CALLABLE void inline mac(uint64_t& ret, uint64_t& carry, const uint64_t a, const uint64_t b,
                              const uint64_t c) noexcept {
  uint128_t ret_tmp = uint128_t{a} + (uint128_t{b} * uint128_t{c}) + uint128_t{carry};

  ret = bast::narrow_cast<uint64_t>(ret_tmp);
  carry = bast::narrow_cast<uint64_t>((ret_tmp >> 64));
}

//--------------------------------------------------------------------------------------------------
// adc
//--------------------------------------------------------------------------------------------------
/*
 Compute a + b + carry, returning the result and the new carry over.
 */
CUDA_CALLABLE void inline adc(uint64_t& ret, uint64_t& carry, const uint64_t a, const uint64_t b,
                              const uint64_t c) noexcept {
  uint128_t ret_tmp = uint128_t{a} + uint128_t{b} + uint128_t{c};

  ret = bast::narrow_cast<uint64_t>(ret_tmp);
  carry = bast::narrow_cast<uint64_t>(ret_tmp >> 64);
}

//--------------------------------------------------------------------------------------------------
// sbb
//--------------------------------------------------------------------------------------------------
/*
 Compute a - (b + borrow), returning the result and the new borrow.
 */
CUDA_CALLABLE void inline sbb(uint64_t& ret, uint64_t& borrow, const uint64_t a,
                              const uint64_t b) noexcept {
  uint128_t ret_tmp = uint128_t{a} - (uint128_t{b} + uint128_t{(borrow >> 63)});

  ret = bast::narrow_cast<uint64_t>(ret_tmp);
  borrow = bast::narrow_cast<uint64_t>(ret_tmp >> 64);
}
} // namespace sxt::basf
