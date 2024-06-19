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
#include "sxt/fieldgk/operation/invert.h"

#include "sxt/fieldgk/operation/pow_vartime.h"
#include "sxt/fieldgk/property/zero.h"
#include "sxt/fieldgk/type/element.h"

namespace sxt::fgko {
//--------------------------------------------------------------------------------------------------
// invert
//--------------------------------------------------------------------------------------------------
/**
 * Computes the multiplicative inverse of this field element,
 * returning FALSE in the case that this element is zero.
 * A finite field of order p is a cyclic group of order p-1.
 * Therefore, for any f in Fp: f^{-1} == f^{p-2}.
 */
CUDA_CALLABLE bool invert(fgkt::element& h, const fgkt::element& f) noexcept {
  constexpr fgkt::element p_v_minus_2{0x3c208c16d87cfd45, 0x97816a916871ca8d, 0xb85045b68181585d,
                                      0x30644e72e131a029};

  fgko::pow_vartime(h, f, p_v_minus_2);

  return fgkp::is_zero(h);
}
} // namespace sxt::fgko
