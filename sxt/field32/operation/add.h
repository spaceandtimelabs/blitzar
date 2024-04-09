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

#include <cstdint>

#include "sxt/base/field/arithmetic_utility.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/field32/base/constants.h"
#include "sxt/field32/base/subtract_p.h"
#include "sxt/field32/type/element.h"

namespace sxt::f32o {
//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE inline void add(f32t::element& h, const f32t::element& f,
                              const f32t::element& g) noexcept {
  uint32_t h_tmp[8] = {};
  uint32_t carry{0};

  basfld::adc(h_tmp[0], carry, f[0], g[0], carry);
  basfld::adc(h_tmp[1], carry, f[1], g[1], carry);
  basfld::adc(h_tmp[2], carry, f[2], g[2], carry);
  basfld::adc(h_tmp[3], carry, f[3], g[3], carry);
  basfld::adc(h_tmp[4], carry, f[4], g[4], carry);
  basfld::adc(h_tmp[5], carry, f[5], g[5], carry);
  basfld::adc(h_tmp[6], carry, f[6], g[6], carry);
  basfld::adc(h_tmp[7], carry, f[7], g[7], carry);

  f32b::subtract_p(h.data(), h_tmp);
}
} // namespace sxt::f32o
