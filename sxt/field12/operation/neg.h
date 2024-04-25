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
#include "sxt/field12/base/constants.h"
#include "sxt/field12/type/element.h"

namespace sxt::f12o {
//--------------------------------------------------------------------------------------------------
// neg
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
inline void neg(f12t::element& h, const f12t::element& f) noexcept {
  uint64_t d[6] = {};
  uint64_t borrow{0};

  basfld::sbb(d[0], borrow, f12b::p_v[0], f[0]);
  basfld::sbb(d[1], borrow, f12b::p_v[1], f[1]);
  basfld::sbb(d[2], borrow, f12b::p_v[2], f[2]);
  basfld::sbb(d[3], borrow, f12b::p_v[3], f[3]);
  basfld::sbb(d[4], borrow, f12b::p_v[4], f[4]);
  basfld::sbb(d[5], borrow, f12b::p_v[5], f[5]);

  // Let's use a mask if `self` was zero, which would mean
  // the result of the subtraction is p.
  uint64_t mask = uint64_t{((f[0] | f[1] | f[2] | f[3] | f[4] | f[5]) == 0)} - uint64_t{1};

  h[0] = d[0] & mask;
  h[1] = d[1] & mask;
  h[2] = d[2] & mask;
  h[3] = d[3] & mask;
  h[4] = d[4] & mask;
  h[5] = d[5] & mask;
}
} // namespace sxt::f12o
