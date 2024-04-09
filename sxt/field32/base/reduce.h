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

#include "sxt/base/field/arithmetic_utility.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/field32/base/constants.h"
#include "sxt/field32/base/subtract_p.h"

namespace sxt::f32b {
//--------------------------------------------------------------------------------------------------
// reduce
//--------------------------------------------------------------------------------------------------
/**
 * The Montgomery reduction here is based on Algorithm 14.32 in
 * Handbook of Applied Cryptography
 * <http://cacr.uwaterloo.ca/hac/about/chap14.pdf>.
 */
CUDA_CALLABLE inline void reduce(uint32_t h[8], const uint32_t t[16]) noexcept {
  uint32_t tmp = 0;
  uint32_t carry = 0;
  uint32_t ret[16];

  uint32_t k = t[0] * inv_v;
  basfld::mac(tmp, carry, t[0], k, p_v[0]);
  basfld::mac(ret[1], carry, t[1], k, p_v[1]);
  basfld::mac(ret[2], carry, t[2], k, p_v[2]);
  basfld::mac(ret[3], carry, t[3], k, p_v[3]);
  basfld::mac(ret[4], carry, t[4], k, p_v[4]);
  basfld::mac(ret[5], carry, t[5], k, p_v[5]);
  basfld::mac(ret[6], carry, t[6], k, p_v[6]);
  basfld::mac(ret[7], carry, t[7], k, p_v[7]);
  basfld::adc(ret[8], ret[9], t[8], 0, carry);

  carry = 0;
  k = ret[1] * inv_v;
  basfld::mac(tmp, carry, ret[1], k, p_v[0]);
  basfld::mac(ret[2], carry, ret[2], k, p_v[1]);
  basfld::mac(ret[3], carry, ret[3], k, p_v[2]);
  basfld::mac(ret[4], carry, ret[4], k, p_v[3]);
  basfld::mac(ret[5], carry, ret[5], k, p_v[4]);
  basfld::mac(ret[6], carry, ret[6], k, p_v[5]);
  basfld::mac(ret[7], carry, ret[7], k, p_v[6]);
  basfld::mac(ret[8], carry, ret[8], k, p_v[7]);
  basfld::adc(ret[9], ret[10], t[9], ret[9], carry);

  carry = 0;
  k = ret[2] * inv_v;
  basfld::mac(tmp, carry, ret[2], k, p_v[0]);
  basfld::mac(ret[3], carry, ret[3], k, p_v[1]);
  basfld::mac(ret[4], carry, ret[4], k, p_v[2]);
  basfld::mac(ret[5], carry, ret[5], k, p_v[3]);
  basfld::mac(ret[6], carry, ret[6], k, p_v[4]);
  basfld::mac(ret[7], carry, ret[7], k, p_v[5]);
  basfld::mac(ret[8], carry, ret[8], k, p_v[6]);
  basfld::mac(ret[9], carry, ret[9], k, p_v[7]);
  basfld::adc(ret[10], ret[11], t[10], ret[10], carry);

  carry = 0;
  k = ret[3] * inv_v;
  basfld::mac(tmp, carry, ret[3], k, p_v[0]);
  basfld::mac(ret[4], carry, ret[4], k, p_v[1]);
  basfld::mac(ret[5], carry, ret[5], k, p_v[2]);
  basfld::mac(ret[6], carry, ret[6], k, p_v[3]);
  basfld::mac(ret[7], carry, ret[7], k, p_v[4]);
  basfld::mac(ret[8], carry, ret[8], k, p_v[5]);
  basfld::mac(ret[9], carry, ret[9], k, p_v[6]);
  basfld::mac(ret[10], carry, ret[10], k, p_v[7]);
  basfld::adc(ret[11], ret[12], t[11], ret[11], carry);

  carry = 0;
  k = ret[4] * inv_v;
  basfld::mac(tmp, carry, ret[4], k, p_v[0]);
  basfld::mac(ret[5], carry, ret[5], k, p_v[1]);
  basfld::mac(ret[6], carry, ret[6], k, p_v[2]);
  basfld::mac(ret[7], carry, ret[7], k, p_v[3]);
  basfld::mac(ret[8], carry, ret[8], k, p_v[4]);
  basfld::mac(ret[9], carry, ret[9], k, p_v[5]);
  basfld::mac(ret[10], carry, ret[10], k, p_v[6]);
  basfld::mac(ret[11], carry, ret[11], k, p_v[7]);
  basfld::adc(ret[12], ret[13], t[12], ret[12], carry);

  carry = 0;
  k = ret[5] * inv_v;
  basfld::mac(tmp, carry, ret[5], k, p_v[0]);
  basfld::mac(ret[6], carry, ret[6], k, p_v[1]);
  basfld::mac(ret[7], carry, ret[7], k, p_v[2]);
  basfld::mac(ret[8], carry, ret[8], k, p_v[3]);
  basfld::mac(ret[9], carry, ret[9], k, p_v[4]);
  basfld::mac(ret[10], carry, ret[10], k, p_v[5]);
  basfld::mac(ret[11], carry, ret[11], k, p_v[6]);
  basfld::mac(ret[12], carry, ret[12], k, p_v[7]);
  basfld::adc(ret[13], ret[14], t[13], ret[13], carry);

  carry = 0;
  k = ret[6] * inv_v;
  basfld::mac(tmp, carry, ret[6], k, p_v[0]);
  basfld::mac(ret[7], carry, ret[7], k, p_v[1]);
  basfld::mac(ret[8], carry, ret[8], k, p_v[2]);
  basfld::mac(ret[9], carry, ret[9], k, p_v[3]);
  basfld::mac(ret[10], carry, ret[10], k, p_v[4]);
  basfld::mac(ret[11], carry, ret[11], k, p_v[5]);
  basfld::mac(ret[12], carry, ret[12], k, p_v[6]);
  basfld::mac(ret[13], carry, ret[13], k, p_v[7]);
  basfld::adc(ret[14], ret[15], t[14], ret[14], carry);

  carry = 0;
  k = ret[7] * inv_v;
  basfld::mac(tmp, carry, ret[7], k, p_v[0]);
  basfld::mac(ret[8], carry, ret[8], k, p_v[1]);
  basfld::mac(ret[9], carry, ret[9], k, p_v[2]);
  basfld::mac(ret[10], carry, ret[10], k, p_v[3]);
  basfld::mac(ret[11], carry, ret[11], k, p_v[4]);
  basfld::mac(ret[12], carry, ret[12], k, p_v[5]);
  basfld::mac(ret[13], carry, ret[13], k, p_v[6]);
  basfld::mac(ret[14], carry, ret[14], k, p_v[7]);
  basfld::adc(ret[15], tmp, t[15], ret[15], carry);

  // Attempt to subtract the modulus,
  // to ensure the value is smaller than the modulus.
  uint32_t a[8] = {ret[8], ret[9], ret[10], ret[11], ret[12], ret[13], ret[14], ret[15]};
  subtract_p(h, a);
}

//--------------------------------------------------------------------------------------------------
// is_below_modulus
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE bool is_below_modulus(const uint32_t h[8]) noexcept;
} // namespace sxt::f32b
