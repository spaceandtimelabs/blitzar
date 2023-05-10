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

#include "sxt/base/container/span.h"
#include "sxt/base/macro/cuda_callable.h"

namespace sxt::s25t {
class element;
}

namespace sxt::s25o {
//--------------------------------------------------------------------------------------------------
// inv
//--------------------------------------------------------------------------------------------------
//
// Input:
//   a[0]+256*a[1]+...+256^31*a[31] = s
//
// Output:
//   s_inv = s_inv[0]+256*s_inv[1]+...+256^31*s_inv[31] where (s * s_inv) = 1 % l
//
// where l = 2^252 + 27742317777372353535851937790883648493
CUDA_CALLABLE
void inv(s25t::element& s_inv, const s25t::element& s) noexcept;

//--------------------------------------------------------------------------------------------------
// batch_inv
//--------------------------------------------------------------------------------------------------
void batch_inv(basct::span<s25t::element> sx_inv, basct::cspan<s25t::element> sx) noexcept;
} // namespace sxt::s25o
