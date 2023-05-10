/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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
#pragma once

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/field51/operation/cmov.h"
#include "sxt/field51/operation/neg.h"
#include "sxt/field51/type/element.h"

namespace sxt::f51o {
//--------------------------------------------------------------------------------------------------
// cneg
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
inline void cneg(f51t::element& h, const f51t::element& f, unsigned int b) noexcept {
  f51t::element negf;

  f51o::neg(negf, f);
  h = f;
  f51o::cmov(h, negf, b);
}
} // namespace sxt::f51o
