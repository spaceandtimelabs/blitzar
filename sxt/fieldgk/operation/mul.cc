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
#include "sxt/fieldgk/operation/mul.h"

#include "sxt/base/field/arithmetic_utility.h"
#include "sxt/fieldgk/base/reduce.h"
#include "sxt/fieldgk/type/element.h"

namespace sxt::fgko {
//--------------------------------------------------------------------------------------------------
// mul
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void mul(fgkt::element& h, const fgkt::element& f, const fgkt::element& g) noexcept {
  uint64_t t[8] = {};
  uint64_t carry{0};

  basfld::mac(t[0], carry, 0, f[0], g[0]);
  basfld::mac(t[1], carry, 0, f[0], g[1]);
  basfld::mac(t[2], carry, 0, f[0], g[2]);
  basfld::mac(t[3], carry, 0, f[0], g[3]);
  t[4] = carry;
  carry = 0;

  basfld::mac(t[1], carry, t[1], f[1], g[0]);
  basfld::mac(t[2], carry, t[2], f[1], g[1]);
  basfld::mac(t[3], carry, t[3], f[1], g[2]);
  basfld::mac(t[4], carry, t[4], f[1], g[3]);
  t[5] = carry;
  carry = 0;

  basfld::mac(t[2], carry, t[2], f[2], g[0]);
  basfld::mac(t[3], carry, t[3], f[2], g[1]);
  basfld::mac(t[4], carry, t[4], f[2], g[2]);
  basfld::mac(t[5], carry, t[5], f[2], g[3]);
  t[6] = carry;
  carry = 0;

  basfld::mac(t[3], carry, t[3], f[3], g[0]);
  basfld::mac(t[4], carry, t[4], f[3], g[1]);
  basfld::mac(t[5], carry, t[5], f[3], g[2]);
  basfld::mac(t[6], carry, t[6], f[3], g[3]);
  t[7] = carry;

  fgkb::reduce(h.data(), t);
}
} // namespace sxt::fgko
