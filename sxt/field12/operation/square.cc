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
#include "sxt/field12/operation/square.h"

#include "sxt/base/field/arithmetic_utility.h"
#include "sxt/field12/base/reduce.h"
#include "sxt/field12/type/element.h"

namespace sxt::f12o {
//--------------------------------------------------------------------------------------------------
// square
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void square(f12t::element& h, const f12t::element& f) noexcept {
  uint64_t t[12] = {};
  uint64_t carry{0};

  basf::mac(t[1], carry, 0, f[0], f[1]);
  basf::mac(t[2], carry, 0, f[0], f[2]);
  basf::mac(t[3], carry, 0, f[0], f[3]);
  basf::mac(t[4], carry, 0, f[0], f[4]);
  basf::mac(t[5], carry, 0, f[0], f[5]);
  t[6] = carry;
  carry = 0;

  basf::mac(t[3], carry, t[3], f[1], f[2]);
  basf::mac(t[4], carry, t[4], f[1], f[3]);
  basf::mac(t[5], carry, t[5], f[1], f[4]);
  basf::mac(t[6], carry, t[6], f[1], f[5]);
  t[7] = carry;
  carry = 0;

  basf::mac(t[5], carry, t[5], f[2], f[3]);
  basf::mac(t[6], carry, t[6], f[2], f[4]);
  basf::mac(t[7], carry, t[7], f[2], f[5]);
  t[8] = carry;
  carry = 0;

  basf::mac(t[7], carry, t[7], f[3], f[4]);
  basf::mac(t[8], carry, t[8], f[3], f[5]);
  t[9] = carry;
  carry = 0;

  basf::mac(t[9], carry, t[9], f[4], f[5]);
  t[10] = carry;
  carry = 0;

  t[11] = t[10] >> 63;
  t[10] = (t[10] << 1) | (t[9] >> 63);
  t[9] = (t[9] << 1) | (t[8] >> 63);
  t[8] = (t[8] << 1) | (t[7] >> 63);
  t[7] = (t[7] << 1) | (t[6] >> 63);
  t[6] = (t[6] << 1) | (t[5] >> 63);
  t[5] = (t[5] << 1) | (t[4] >> 63);
  t[4] = (t[4] << 1) | (t[3] >> 63);
  t[3] = (t[3] << 1) | (t[2] >> 63);
  t[2] = (t[2] << 1) | (t[1] >> 63);
  t[1] = t[1] << 1;

  basf::mac(t[0], carry, 0, f[0], f[0]);
  basf::adc(t[1], carry, t[1], 0, carry);
  basf::mac(t[2], carry, t[2], f[1], f[1]);
  basf::adc(t[3], carry, t[3], 0, carry);
  basf::mac(t[4], carry, t[4], f[2], f[2]);
  basf::adc(t[5], carry, t[5], 0, carry);
  basf::mac(t[6], carry, t[6], f[3], f[3]);
  basf::adc(t[7], carry, t[7], 0, carry);
  basf::mac(t[8], carry, t[8], f[4], f[4]);
  basf::adc(t[9], carry, t[9], 0, carry);
  basf::mac(t[10], carry, t[10], f[5], f[5]);
  basf::adc(t[11], carry, t[11], 0, carry);

  f12b::reduce(h.data(), t);
}
} // namespace sxt::f12o
