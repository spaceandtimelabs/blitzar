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
#include "sxt/scalar25/type/element.h"

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
//
// Input:
//   x[0]+256*x[1]+...+256^31*x[31] = x
//   y[0]+256*y[1]+...+256^31*y[31] = y
//
// Output:
//   z[0]+256*z[1]+...+256^31*z[31] = (x + y) mod l
//
// where l = 2^252 + 27742317777372353535851937790883648493
CUDA_CALLABLE
void add(s25t::element& z, const s25t::element& x, const s25t::element& y) noexcept;

CUDA_CALLABLE
void add(s25t::element& z, const volatile s25t::element& x,
         const volatile s25t::element& y) noexcept;
} // namespace sxt::s25o
