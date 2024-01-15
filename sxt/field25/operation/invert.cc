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
#include "sxt/field25/operation/invert.h"

#include "sxt/field25/operation/pow_vartime.h"
#include "sxt/field25/property/zero.h"
#include "sxt/field25/type/element.h"

namespace sxt::f25o {
//--------------------------------------------------------------------------------------------------
// invert
//--------------------------------------------------------------------------------------------------
/**
 * Computes the multiplicative inverse of this field element,
 * returning FALSE in the case that this element is zero.
 */
CUDA_CALLABLE bool invert(f12t::element& h, const f12t::element& f) noexcept {
  constexpr f12t::element g(0xb9feffffffffaaa9, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624,
                            0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a);

  f12o::pow_vartime(h, f, g);

  return f12p::is_zero(h);
}
} // namespace sxt::f25o
