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
 * Copyright (c) 2021
 * Sean Bowe <ewillbefull@gmail.com>
 * Jack Grigg <thestr4d@gmail.com>
 * See third_party/license/zkcrypto.LICENSE
 *
 * Adopted from ingonyama-zk/icicle
 * Copyright (c) 2023
 * See third_party/license/ingonyama-zk.LICENSE
 */
#pragma once

#include <cstdint>

#include "sxt/base/field/arithmetic_utility.h"
#include "sxt/base/field/ptx.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/field25/base/constants.h"
#include "sxt/field25/base/subtract_p.h"
#include "sxt/field25/type/element.h"

namespace sxt::f25o {
//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE inline void add(uint64_t* h, const uint64_t* f, const uint64_t* g) noexcept {
  uint64_t h_tmp[4] = {};

#ifdef __CUDA_ARCH__
  h_tmp[0] = basfld::add_cc(f[0], g[0]);
  h_tmp[1] = basfld::addc_cc(f[1], g[1]);
  h_tmp[2] = basfld::addc_cc(f[2], g[2]);
  h_tmp[3] = basfld::addc(f[3], g[3]);
#else
  uint64_t carry{0};

  basfld::adc(h_tmp[0], carry, f[0], g[0], carry);
  basfld::adc(h_tmp[1], carry, f[1], g[1], carry);
  basfld::adc(h_tmp[2], carry, f[2], g[2], carry);
  basfld::adc(h_tmp[3], carry, f[3], g[3], carry);
#endif

  f25b::subtract_p(h, h_tmp);
}

//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE inline void add(f25t::element& h, const f25t::element& f,
                              const f25t::element& g) noexcept {
  add(h.data(), f.data(), g.data());
}
} // namespace sxt::f25o
