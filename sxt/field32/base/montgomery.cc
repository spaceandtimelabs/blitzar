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
#include "sxt/field32/base/montgomery.h"

#include "sxt/base/field/arithmetic_utility.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/field32/base/constants.h"
#include "sxt/field32/base/reduce.h"

namespace sxt::f32b {
//--------------------------------------------------------------------------------------------------
// to_montgomery_form
//-------------------------------------------------------------------------------------------------
CUDA_CALLABLE void to_montgomery_form(uint32_t h[8], const uint32_t s[8]) noexcept {
  uint32_t t[16] = {};
  uint32_t carry = 0;

  basfld::mac(t[0], carry, 0, s[0], r2_v[0]);
  basfld::mac(t[1], carry, 0, s[0], r2_v[1]);
  basfld::mac(t[2], carry, 0, s[0], r2_v[2]);
  basfld::mac(t[3], carry, 0, s[0], r2_v[3]);
  basfld::mac(t[4], carry, 0, s[0], r2_v[4]);
  basfld::mac(t[5], carry, 0, s[0], r2_v[5]);
  basfld::mac(t[6], carry, 0, s[0], r2_v[6]);
  basfld::mac(t[7], carry, 0, s[0], r2_v[7]);
  t[8] = carry;
  carry = 0;

  basfld::mac(t[1], carry, t[1], s[1], r2_v[0]);
  basfld::mac(t[2], carry, t[2], s[1], r2_v[1]);
  basfld::mac(t[3], carry, t[3], s[1], r2_v[2]);
  basfld::mac(t[4], carry, t[4], s[1], r2_v[3]);
  basfld::mac(t[5], carry, t[5], s[1], r2_v[4]);
  basfld::mac(t[6], carry, t[6], s[1], r2_v[5]);
  basfld::mac(t[7], carry, t[7], s[1], r2_v[6]);
  basfld::mac(t[8], carry, t[8], s[1], r2_v[7]);
  t[9] = carry;
  carry = 0;

  basfld::mac(t[2], carry, t[2], s[2], r2_v[0]);
  basfld::mac(t[3], carry, t[3], s[2], r2_v[1]);
  basfld::mac(t[4], carry, t[4], s[2], r2_v[2]);
  basfld::mac(t[5], carry, t[5], s[2], r2_v[3]);
  basfld::mac(t[6], carry, t[6], s[2], r2_v[4]);
  basfld::mac(t[7], carry, t[7], s[2], r2_v[5]);
  basfld::mac(t[8], carry, t[8], s[2], r2_v[6]);
  basfld::mac(t[9], carry, t[9], s[2], r2_v[7]);
  t[10] = carry;
  carry = 0;

  basfld::mac(t[3], carry, t[3], s[3], r2_v[0]);
  basfld::mac(t[4], carry, t[4], s[3], r2_v[1]);
  basfld::mac(t[5], carry, t[5], s[3], r2_v[2]);
  basfld::mac(t[6], carry, t[6], s[3], r2_v[3]);
  basfld::mac(t[7], carry, t[7], s[3], r2_v[4]);
  basfld::mac(t[8], carry, t[8], s[3], r2_v[5]);
  basfld::mac(t[9], carry, t[9], s[3], r2_v[6]);
  basfld::mac(t[10], carry, t[10], s[3], r2_v[7]);
  t[11] = carry;
  carry = 0;

  basfld::mac(t[4], carry, t[4], s[4], r2_v[0]);
  basfld::mac(t[5], carry, t[5], s[4], r2_v[1]);
  basfld::mac(t[6], carry, t[6], s[4], r2_v[2]);
  basfld::mac(t[7], carry, t[7], s[4], r2_v[3]);
  basfld::mac(t[8], carry, t[8], s[4], r2_v[4]);
  basfld::mac(t[9], carry, t[9], s[4], r2_v[5]);
  basfld::mac(t[10], carry, t[10], s[4], r2_v[6]);
  basfld::mac(t[11], carry, t[11], s[4], r2_v[7]);
  t[12] = carry;
  carry = 0;

  basfld::mac(t[5], carry, t[5], s[5], r2_v[0]);
  basfld::mac(t[6], carry, t[6], s[5], r2_v[1]);
  basfld::mac(t[7], carry, t[7], s[5], r2_v[2]);
  basfld::mac(t[8], carry, t[8], s[5], r2_v[3]);
  basfld::mac(t[9], carry, t[9], s[5], r2_v[4]);
  basfld::mac(t[10], carry, t[10], s[5], r2_v[5]);
  basfld::mac(t[11], carry, t[11], s[5], r2_v[6]);
  basfld::mac(t[12], carry, t[12], s[5], r2_v[7]);
  t[13] = carry;
  carry = 0;

  basfld::mac(t[6], carry, t[6], s[6], r2_v[0]);
  basfld::mac(t[7], carry, t[7], s[6], r2_v[1]);
  basfld::mac(t[8], carry, t[8], s[6], r2_v[2]);
  basfld::mac(t[9], carry, t[9], s[6], r2_v[3]);
  basfld::mac(t[10], carry, t[10], s[6], r2_v[4]);
  basfld::mac(t[11], carry, t[11], s[6], r2_v[5]);
  basfld::mac(t[12], carry, t[12], s[6], r2_v[6]);
  basfld::mac(t[13], carry, t[13], s[6], r2_v[7]);
  t[14] = carry;
  carry = 0;

  basfld::mac(t[7], carry, t[7], s[7], r2_v[0]);
  basfld::mac(t[8], carry, t[8], s[7], r2_v[1]);
  basfld::mac(t[9], carry, t[9], s[7], r2_v[2]);
  basfld::mac(t[10], carry, t[10], s[7], r2_v[3]);
  basfld::mac(t[11], carry, t[11], s[7], r2_v[4]);
  basfld::mac(t[12], carry, t[12], s[7], r2_v[5]);
  basfld::mac(t[13], carry, t[13], s[7], r2_v[6]);
  basfld::mac(t[14], carry, t[14], s[7], r2_v[7]);
  t[15] = carry;

  reduce(h, t);
}
} // namespace sxt::f32b
