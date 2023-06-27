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
#include "sxt/field12/base/byte_conversion.h"

#include "sxt/base/bit/load.h"
#include "sxt/base/bit/store.h"
#include "sxt/field12/base/montgomery.h"
#include "sxt/field12/base/reduce.h"

namespace sxt::f12b {
//--------------------------------------------------------------------------------------------------
// from_bytes
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void from_bytes(bool& below_modulus, uint64_t h[6], const uint8_t s[48]) noexcept {
  h[0] = (basbt::load64_le(s));
  h[1] = (basbt::load64_le(s + 8));
  h[2] = (basbt::load64_le(s + 16));
  h[3] = (basbt::load64_le(s + 24));
  h[4] = (basbt::load64_le(s + 32));
  h[5] = (basbt::load64_le(s + 40));

  below_modulus = is_below_modulus(h);

  // Convert to Montgomery form by computing
  // (a.R^0 * R^2) / R = a.R
  to_montgomery_form(h, h);
}

//--------------------------------------------------------------------------------------------------
// to_bytes
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void to_bytes(uint8_t s[48], const uint64_t h[6]) noexcept {
  // Turn into canonical form by computing
  // (a.R) / R = a
  uint64_t t[12] = {h[0], h[1], h[2], h[3], h[4], h[5], 0, 0, 0, 0, 0, 0};
  uint64_t h_tmp[6] = {};
  reduce(h_tmp, t);

  basbt::store64_le(s + 0, h_tmp[0]);
  basbt::store64_le(s + 8, h_tmp[1]);
  basbt::store64_le(s + 16, h_tmp[2]);
  basbt::store64_le(s + 24, h_tmp[3]);
  basbt::store64_le(s + 32, h_tmp[4]);
  basbt::store64_le(s + 40, h_tmp[5]);
}
} // namespace sxt::f12b
