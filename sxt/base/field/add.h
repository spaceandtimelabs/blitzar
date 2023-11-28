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
#pragma once

#include <cstdint>

#include "sxt/base/field/arithmetic_utility.h"
#include "sxt/base/field/element.h"
#include "sxt/base/field/subtract_p.h"
#include "sxt/base/macro/cuda_callable.h"

namespace sxt::basfld {
//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
template <basfld::element Element>
CUDA_CALLABLE inline void add(Element& h, const Element& f, const Element& g) noexcept {
  Element h_tmp;
  uint64_t carry{0};

  for (size_t limb = 0; limb < h.num_limbs_v; ++limb) {
    adc(h_tmp[limb], carry, f[limb], g[limb], carry);
  }

  subtract_p<Element::num_limbs_v>(h.data(), h_tmp.data(), Element::modulus().data());
}
} // namespace sxt::basfld
