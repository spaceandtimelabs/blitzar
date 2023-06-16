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
#include "sxt/field12/base/arithmetic_utility.h"

#include "sxt/base/type/int.h"
#include "sxt/base/type/narrow_cast.h"

namespace sxt::f12b {
//--------------------------------------------------------------------------------------------------
// mac
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE void mac(uint64_t& ret, uint64_t& carry, const uint64_t a, const uint64_t b,
                       const uint64_t c) noexcept {
  uint128_t ret_tmp = uint128_t{a} + (uint128_t{b} * uint128_t{c}) + uint128_t{carry};

  ret = bast::narrow_cast<uint64_t>(ret_tmp);
  carry = bast::narrow_cast<uint64_t>((ret_tmp >> 64));
}

//--------------------------------------------------------------------------------------------------
// adc
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE void adc(uint64_t& ret, uint64_t& carry, const uint64_t a, const uint64_t b,
                       const uint64_t c) noexcept {
  uint128_t ret_tmp = uint128_t{a} + uint128_t{b} + uint128_t{c};

  ret = bast::narrow_cast<uint64_t>(ret_tmp);
  carry = bast::narrow_cast<uint64_t>(ret_tmp >> 64);
}

//--------------------------------------------------------------------------------------------------
// sbb
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE void sbb(uint64_t& ret, uint64_t& borrow, const uint64_t a,
                       const uint64_t b) noexcept {
  uint128_t ret_tmp = uint128_t{a} - (uint128_t{b} + uint128_t{(borrow >> 63)});

  ret = bast::narrow_cast<uint64_t>(ret_tmp);
  borrow = bast::narrow_cast<uint64_t>(ret_tmp >> 64);
}

//--------------------------------------------------------------------------------------------------
// subtract
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE void subtract_p(uint64_t ret[6], const uint64_t a[6],
                              const uint64_t modulus[6]) noexcept {
  uint64_t borrow = 0;
  sbb(ret[0], borrow, a[0], modulus[0]);
  sbb(ret[1], borrow, a[1], modulus[1]);
  sbb(ret[2], borrow, a[2], modulus[2]);
  sbb(ret[3], borrow, a[3], modulus[3]);
  sbb(ret[4], borrow, a[4], modulus[4]);
  sbb(ret[5], borrow, a[5], modulus[5]);

  // If underflow occurred on the final limb, borrow = 0xfff...fff, otherwise
  // borrow = 0x000...000. Thus, we use it as a mask!
  uint64_t mask = borrow == 0x0 ? (borrow - 1) : 0x0;
  ret[0] = (a[0] & borrow) | (ret[0] & mask);
  ret[1] = (a[1] & borrow) | (ret[1] & mask);
  ret[2] = (a[2] & borrow) | (ret[2] & mask);
  ret[3] = (a[3] & borrow) | (ret[3] & mask);
  ret[4] = (a[4] & borrow) | (ret[4] & mask);
  ret[5] = (a[5] & borrow) | (ret[5] & mask);
}
} // namespace sxt::f12b
