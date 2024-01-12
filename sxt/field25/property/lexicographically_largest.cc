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
#include "sxt/field25/property/lexicographically_largest.h"

#include "sxt/base/field/arithmetic_utility.h"
#include "sxt/field25/base/reduce.h"
#include "sxt/field25/type/element.h"

namespace sxt::f25p {
//--------------------------------------------------------------------------------------------------
// lexicographically_largest
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
bool lexicographically_largest(const f25t::element& e) noexcept {
  // Checking to see if the element is larger than (p - 1) / 2.
  // If we subtract by ((p - 1) / 2) + 1 and there is no underflow,
  // then the element must be larger than (p - 1) / 2.

  // First, because e is in Montgomery form we need to reduce it.
  f25t::element e_tmp;
  uint64_t tmp[8]{e[0], e[1], e[2], e[3], 0, 0, 0, 0};
  f25b::reduce(e_tmp.data(), tmp);

  uint64_t dummy{0};
  uint64_t borrow{0};
  basfld::sbb(dummy, borrow, e_tmp[0], 0x9e10460b6c3e7ea4);
  basfld::sbb(dummy, borrow, e_tmp[1], 0xcbc0b548b438e546);
  basfld::sbb(dummy, borrow, e_tmp[2], 0xdc2822db40c0ac2e);
  basfld::sbb(dummy, borrow, e_tmp[3], 0x183227397098d014);

  // If the element was smaller, the subtraction will underflow
  // producing a borrow value of 0xffff...ffff, otherwise it will
  // be zero. We create a Choice representing true if there was
  // overflow (and so this element is not lexicographically larger
  // than its negation) and then negate it.
  return !(borrow & 1);
}
} // namespace sxt::f25p
