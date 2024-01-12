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

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/field12/operation/add.h"
#include "sxt/field12/operation/neg.h"
#include "sxt/field12/type/element.h"

namespace sxt::f12o {
//--------------------------------------------------------------------------------------------------
// sub
//--------------------------------------------------------------------------------------------------
/*
 h = f - g
*/
CUDA_CALLABLE
inline void sub(f12t::element& h, const f12t::element& f, const f12t::element& g) noexcept {
  f12t::element neg_g;
  f12o::neg(neg_g, g);
  f12o::add(h, f, neg_g);
}
} // namespace sxt::f12o
