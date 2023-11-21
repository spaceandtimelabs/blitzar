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
#include "sxt/field12/property/lexicographically_largest.h"

#include "sxt/base/field/arithmetic_utility.h"
#include "sxt/field12/base/reduce.h"
#include "sxt/field12/type/element.h"

namespace sxt::f12p {
//--------------------------------------------------------------------------------------------------
// lexicographically_largest
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
bool lexicographically_largest(const f12t::element& e) noexcept {
  // Checking to see if the element is larger than (p - 1) / 2.
  // If we subtract by ((p - 1) / 2) + 1 and there is no underflow,
  // then the element must be larger than (p - 1) / 2.

  // First, because self is in Montgomery form we need to reduce it.
  f12t::element e_tmp;
  uint64_t tmp[12]{e[0], e[1], e[2], e[3], e[4], e[5], 0, 0, 0, 0, 0, 0};
  f12b::reduce(e_tmp.data(), tmp);

  uint64_t dummy{0};
  uint64_t borrow{0};
  basf::sbb(dummy, borrow, e_tmp[0], 0xdcff7fffffffd556);
  basf::sbb(dummy, borrow, e_tmp[1], 0x0f55ffff58a9ffff);
  basf::sbb(dummy, borrow, e_tmp[2], 0xb39869507b587b12);
  basf::sbb(dummy, borrow, e_tmp[3], 0xb23ba5c279c2895f);
  basf::sbb(dummy, borrow, e_tmp[4], 0x258dd3db21a5d66b);
  basf::sbb(dummy, borrow, e_tmp[5], 0x0d0088f51cbff34d);

  // If the element was smaller, the subtraction will underflow
  // producing a borrow value of 0xffff...ffff, otherwise it will
  // be zero. We create a Choice representing true if there was
  // overflow (and so this element is not lexicographically larger
  // than its negation) and then negate it.
  return !(borrow & 1);
}
} // namespace sxt::f12p
