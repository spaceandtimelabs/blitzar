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
#include "sxt/field32/operation/mul.h"

#include "sxt/base/field/arithmetic_utility.h"
#include "sxt/field32/base/reduce.h"
#include "sxt/field32/type/element.h"

namespace sxt::f32o {
//--------------------------------------------------------------------------------------------------
// mul
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void mul(f32t::element& h, const f32t::element& f, const f32t::element& g) noexcept {
  uint32_t t[16] = {};
  uint32_t carry{0};

  basfld::mac(t[0], carry, 0, f[0], g[0]);
  basfld::mac(t[1], carry, 0, f[0], g[1]);
  basfld::mac(t[2], carry, 0, f[0], g[2]);
  basfld::mac(t[3], carry, 0, f[0], g[3]);
  basfld::mac(t[4], carry, 0, f[0], g[4]);
  basfld::mac(t[5], carry, 0, f[0], g[5]);
  basfld::mac(t[6], carry, 0, f[0], g[6]);
  basfld::mac(t[7], carry, 0, f[0], g[7]);
  t[8] = carry;
  carry = 0;

  basfld::mac(t[1], carry, t[1], f[1], g[0]);
  basfld::mac(t[2], carry, t[2], f[1], g[1]);
  basfld::mac(t[3], carry, t[3], f[1], g[2]);
  basfld::mac(t[4], carry, t[4], f[1], g[3]);
  basfld::mac(t[5], carry, t[5], f[1], g[4]);
  basfld::mac(t[6], carry, t[6], f[1], g[5]);
  basfld::mac(t[7], carry, t[7], f[1], g[6]);
  basfld::mac(t[8], carry, t[8], f[1], g[7]);
  t[9] = carry;
  carry = 0;

  basfld::mac(t[2], carry, t[2], f[2], g[0]);
  basfld::mac(t[3], carry, t[3], f[2], g[1]);
  basfld::mac(t[4], carry, t[4], f[2], g[2]);
  basfld::mac(t[5], carry, t[5], f[2], g[3]);
  basfld::mac(t[6], carry, t[6], f[2], g[4]);
  basfld::mac(t[7], carry, t[7], f[2], g[5]);
  basfld::mac(t[8], carry, t[8], f[2], g[6]);
  basfld::mac(t[9], carry, t[9], f[2], g[7]);
  t[10] = carry;
  carry = 0;

  basfld::mac(t[3], carry, t[3], f[3], g[0]);
  basfld::mac(t[4], carry, t[4], f[3], g[1]);
  basfld::mac(t[5], carry, t[5], f[3], g[2]);
  basfld::mac(t[6], carry, t[6], f[3], g[3]);
  basfld::mac(t[7], carry, t[7], f[3], g[4]);
  basfld::mac(t[8], carry, t[8], f[3], g[5]);
  basfld::mac(t[9], carry, t[9], f[3], g[6]);
  basfld::mac(t[10], carry, t[10], f[3], g[7]);
  t[11] = carry;
  carry = 0;

  basfld::mac(t[4], carry, t[4], f[4], g[0]);
  basfld::mac(t[5], carry, t[5], f[4], g[1]);
  basfld::mac(t[6], carry, t[6], f[4], g[2]);
  basfld::mac(t[7], carry, t[7], f[4], g[3]);
  basfld::mac(t[8], carry, t[8], f[4], g[4]);
  basfld::mac(t[9], carry, t[9], f[4], g[5]);
  basfld::mac(t[10], carry, t[10], f[4], g[6]);
  basfld::mac(t[11], carry, t[11], f[4], g[7]);
  t[12] = carry;
  carry = 0;

  basfld::mac(t[5], carry, t[5], f[5], g[0]);
  basfld::mac(t[6], carry, t[6], f[5], g[1]);
  basfld::mac(t[7], carry, t[7], f[5], g[2]);
  basfld::mac(t[8], carry, t[8], f[5], g[3]);
  basfld::mac(t[9], carry, t[9], f[5], g[4]);
  basfld::mac(t[10], carry, t[10], f[5], g[5]);
  basfld::mac(t[11], carry, t[11], f[5], g[6]);
  basfld::mac(t[12], carry, t[12], f[5], g[7]);
  t[13] = carry;
  carry = 0;

  basfld::mac(t[6], carry, t[6], f[6], g[0]);
  basfld::mac(t[7], carry, t[7], f[6], g[1]);
  basfld::mac(t[8], carry, t[8], f[6], g[2]);
  basfld::mac(t[9], carry, t[9], f[6], g[3]);
  basfld::mac(t[10], carry, t[10], f[6], g[4]);
  basfld::mac(t[11], carry, t[11], f[6], g[5]);
  basfld::mac(t[12], carry, t[12], f[6], g[6]);
  basfld::mac(t[13], carry, t[13], f[6], g[7]);
  t[14] = carry;
  carry = 0;

  basfld::mac(t[7], carry, t[7], f[7], g[0]);
  basfld::mac(t[8], carry, t[8], f[7], g[1]);
  basfld::mac(t[9], carry, t[9], f[7], g[2]);
  basfld::mac(t[10], carry, t[10], f[7], g[3]);
  basfld::mac(t[11], carry, t[11], f[7], g[4]);
  basfld::mac(t[12], carry, t[12], f[7], g[5]);
  basfld::mac(t[13], carry, t[13], f[7], g[6]);
  basfld::mac(t[14], carry, t[14], f[7], g[7]);
  t[15] = carry;

  f32b::reduce(h.data(), t);
}
} // namespace sxt::f32o
