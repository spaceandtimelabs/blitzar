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
#include "sxt/field12/base/montgomery.h"

#include "sxt/base/field/arithmetic_utility.h"
#include "sxt/field12/base/constants.h"
#include "sxt/field12/base/reduce.h"

namespace sxt::f12b {
//--------------------------------------------------------------------------------------------------
// to_montgomery_form
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE void to_montgomery_form(uint64_t h[6], const uint64_t s[6]) noexcept {
  uint64_t t[12] = {};
  uint64_t carry = 0;

  basfld::mac(t[0], carry, 0, s[0], r2_v[0]);
  basfld::mac(t[1], carry, 0, s[0], r2_v[1]);
  basfld::mac(t[2], carry, 0, s[0], r2_v[2]);
  basfld::mac(t[3], carry, 0, s[0], r2_v[3]);
  basfld::mac(t[4], carry, 0, s[0], r2_v[4]);
  basfld::mac(t[5], carry, 0, s[0], r2_v[5]);
  t[6] = carry;
  carry = 0;

  basfld::mac(t[1], carry, t[1], s[1], r2_v[0]);
  basfld::mac(t[2], carry, t[2], s[1], r2_v[1]);
  basfld::mac(t[3], carry, t[3], s[1], r2_v[2]);
  basfld::mac(t[4], carry, t[4], s[1], r2_v[3]);
  basfld::mac(t[5], carry, t[5], s[1], r2_v[4]);
  basfld::mac(t[6], carry, t[6], s[1], r2_v[5]);
  t[7] = carry;
  carry = 0;

  basfld::mac(t[2], carry, t[2], s[2], r2_v[0]);
  basfld::mac(t[3], carry, t[3], s[2], r2_v[1]);
  basfld::mac(t[4], carry, t[4], s[2], r2_v[2]);
  basfld::mac(t[5], carry, t[5], s[2], r2_v[3]);
  basfld::mac(t[6], carry, t[6], s[2], r2_v[4]);
  basfld::mac(t[7], carry, t[7], s[2], r2_v[5]);
  t[8] = carry;
  carry = 0;

  basfld::mac(t[3], carry, t[3], s[3], r2_v[0]);
  basfld::mac(t[4], carry, t[4], s[3], r2_v[1]);
  basfld::mac(t[5], carry, t[5], s[3], r2_v[2]);
  basfld::mac(t[6], carry, t[6], s[3], r2_v[3]);
  basfld::mac(t[7], carry, t[7], s[3], r2_v[4]);
  basfld::mac(t[8], carry, t[8], s[3], r2_v[5]);
  t[9] = carry;
  carry = 0;

  basfld::mac(t[4], carry, t[4], s[4], r2_v[0]);
  basfld::mac(t[5], carry, t[5], s[4], r2_v[1]);
  basfld::mac(t[6], carry, t[6], s[4], r2_v[2]);
  basfld::mac(t[7], carry, t[7], s[4], r2_v[3]);
  basfld::mac(t[8], carry, t[8], s[4], r2_v[4]);
  basfld::mac(t[9], carry, t[9], s[4], r2_v[5]);
  t[10] = carry;
  carry = 0;

  basfld::mac(t[5], carry, t[5], s[5], r2_v[0]);
  basfld::mac(t[6], carry, t[6], s[5], r2_v[1]);
  basfld::mac(t[7], carry, t[7], s[5], r2_v[2]);
  basfld::mac(t[8], carry, t[8], s[5], r2_v[3]);
  basfld::mac(t[9], carry, t[9], s[5], r2_v[4]);
  basfld::mac(t[10], carry, t[10], s[5], r2_v[5]);
  t[11] = carry;

  reduce(h, t);
}
} // namespace sxt::f12b
