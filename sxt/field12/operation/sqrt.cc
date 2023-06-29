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
#include "sxt/field12/operation/sqrt.h"

#include "sxt/field12/operation/pow_vartime.h"
#include "sxt/field12/operation/square.h"
#include "sxt/field12/type/element.h"

namespace sxt::f12o {
//--------------------------------------------------------------------------------------------------
// sqrt
//--------------------------------------------------------------------------------------------------
/*
 We use Shank's method, as p = 3 (mod 4). This means we only need to exponentiate by (p+1)/4.
 This only works for elements that are actually quadratic residue so we check that we got the
 correct result at the end and return a boolean value indicating if the result was correct.
*/
CUDA_CALLABLE bool sqrt(f12t::element& h, const f12t::element& f) noexcept {
  constexpr f12t::element g(0xee7fbfffffffeaab, 0x07aaffffac54ffff, 0xd9cc34a83dac3d89,
                            0xd91dd2e13ce144af, 0x92c6e9ed90d2eb35, 0x0680447a8e5ff9a6);

  f12o::pow_vartime(h, f, g);

  f12t::element h_sq;
  f12o::square(h_sq, h);
  return h_sq == f;
}
} // namespace sxt::f12o
