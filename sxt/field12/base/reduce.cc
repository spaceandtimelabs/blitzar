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
#include "sxt/field12/base/reduce.h"

#include "sxt/base/field/arithmetic_utility.h"
#include "sxt/base/type/narrow_cast.h"
#include "sxt/field12/base/constants.h"
#include "sxt/field12/base/subtract_p.h"

namespace sxt::f12b {
CUDA_CALLABLE void reduce(uint64_t h[6], const uint64_t t[12]) noexcept {
  uint64_t tmp = 0;
  uint64_t carry = 0;
  uint64_t ret[12];

  uint64_t k = t[0] * inv_v;
  basfld::mac(tmp, carry, t[0], k, p_v[0]);
  basfld::mac(ret[1], carry, t[1], k, p_v[1]);
  basfld::mac(ret[2], carry, t[2], k, p_v[2]);
  basfld::mac(ret[3], carry, t[3], k, p_v[3]);
  basfld::mac(ret[4], carry, t[4], k, p_v[4]);
  basfld::mac(ret[5], carry, t[5], k, p_v[5]);
  basfld::adc(ret[6], ret[7], t[6], 0, carry);

  carry = 0;
  k = ret[1] * inv_v;
  basfld::mac(tmp, carry, ret[1], k, p_v[0]);
  basfld::mac(ret[2], carry, ret[2], k, p_v[1]);
  basfld::mac(ret[3], carry, ret[3], k, p_v[2]);
  basfld::mac(ret[4], carry, ret[4], k, p_v[3]);
  basfld::mac(ret[5], carry, ret[5], k, p_v[4]);
  basfld::mac(ret[6], carry, ret[6], k, p_v[5]);
  basfld::adc(ret[7], ret[8], t[7], ret[7], carry);

  carry = 0;
  k = ret[2] * inv_v;
  basfld::mac(tmp, carry, ret[2], k, p_v[0]);
  basfld::mac(ret[3], carry, ret[3], k, p_v[1]);
  basfld::mac(ret[4], carry, ret[4], k, p_v[2]);
  basfld::mac(ret[5], carry, ret[5], k, p_v[3]);
  basfld::mac(ret[6], carry, ret[6], k, p_v[4]);
  basfld::mac(ret[7], carry, ret[7], k, p_v[5]);
  basfld::adc(ret[8], ret[9], t[8], ret[8], carry);

  carry = 0;
  k = ret[3] * inv_v;
  basfld::mac(tmp, carry, ret[3], k, p_v[0]);
  basfld::mac(ret[4], carry, ret[4], k, p_v[1]);
  basfld::mac(ret[5], carry, ret[5], k, p_v[2]);
  basfld::mac(ret[6], carry, ret[6], k, p_v[3]);
  basfld::mac(ret[7], carry, ret[7], k, p_v[4]);
  basfld::mac(ret[8], carry, ret[8], k, p_v[5]);
  basfld::adc(ret[9], ret[10], t[9], ret[9], carry);

  carry = 0;
  k = ret[4] * inv_v;
  basfld::mac(tmp, carry, ret[4], k, p_v[0]);
  basfld::mac(ret[5], carry, ret[5], k, p_v[1]);
  basfld::mac(ret[6], carry, ret[6], k, p_v[2]);
  basfld::mac(ret[7], carry, ret[7], k, p_v[3]);
  basfld::mac(ret[8], carry, ret[8], k, p_v[4]);
  basfld::mac(ret[9], carry, ret[9], k, p_v[5]);
  basfld::adc(ret[10], ret[11], t[10], ret[10], carry);

  carry = 0;
  k = ret[5] * inv_v;
  basfld::mac(tmp, carry, ret[5], k, p_v[0]);
  basfld::mac(ret[6], carry, ret[6], k, p_v[1]);
  basfld::mac(ret[7], carry, ret[7], k, p_v[2]);
  basfld::mac(ret[8], carry, ret[8], k, p_v[3]);
  basfld::mac(ret[9], carry, ret[9], k, p_v[4]);
  basfld::mac(ret[10], carry, ret[10], k, p_v[5]);
  basfld::adc(ret[11], tmp, t[11], ret[11], carry);

  // Attempt to subtract the modulus,
  // to ensure the value is smaller than the modulus.
  uint64_t a[6] = {ret[6], ret[7], ret[8], ret[9], ret[10], ret[11]};
  subtract_p(h, a);
}
//--------------------------------------------------------------------------------------------------
// is_below_modulus
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE bool is_below_modulus(const uint64_t h[6]) noexcept {
  uint64_t borrow = 0;
  uint64_t ret[6] = {};

  // Try to subtract the modulus
  basfld::sbb(ret[0], borrow, h[0], p_v[0]);
  basfld::sbb(ret[1], borrow, h[1], p_v[1]);
  basfld::sbb(ret[2], borrow, h[2], p_v[2]);
  basfld::sbb(ret[3], borrow, h[3], p_v[3]);
  basfld::sbb(ret[4], borrow, h[4], p_v[4]);
  basfld::sbb(ret[5], borrow, h[5], p_v[5]);

  // If the element is smaller than MODULUS then the
  // subtraction will underflow, producing a borrow value
  // of 0xffff...ffff. Otherwise, it'll be zero.
  return bast::narrow_cast<uint8_t>(borrow) & 1;
}
} // namespace sxt::f12b
