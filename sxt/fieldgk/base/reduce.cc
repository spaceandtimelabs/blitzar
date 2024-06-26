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
#include "sxt/fieldgk/base/reduce.h"

#include "sxt/base/type/narrow_cast.h"

namespace sxt::fgkb {
//--------------------------------------------------------------------------------------------------
// is_below_modulus
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE bool is_below_modulus(const uint64_t h[4]) noexcept {
  uint64_t borrow = 0;
  uint64_t ret[4] = {};

  // Try to subtract the modulus
  basfld::sbb(ret[0], borrow, h[0], p_v[0]);
  basfld::sbb(ret[1], borrow, h[1], p_v[1]);
  basfld::sbb(ret[2], borrow, h[2], p_v[2]);
  basfld::sbb(ret[3], borrow, h[3], p_v[3]);

  // If the element is smaller than MODULUS then the
  // subtraction will underflow, producing a borrow value
  // of 0xffff...ffff. Otherwise, it'll be zero.
  return bast::narrow_cast<uint8_t>(borrow) & 1;
}
} // namespace sxt::fgkb
