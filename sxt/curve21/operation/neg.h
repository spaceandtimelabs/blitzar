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
#pragma once

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::c21t {
struct element_p3;
}

namespace sxt::c21o {
//--------------------------------------------------------------------------------------------------
// neg
//--------------------------------------------------------------------------------------------------
/* r = -p */
CUDA_CALLABLE
void neg(c21t::element_p3& r, const c21t::element_p3& p) noexcept;

//--------------------------------------------------------------------------------------------------
// cneg
//--------------------------------------------------------------------------------------------------
/* r = -r if b = 1 else r */
CUDA_CALLABLE
void cneg(c21t::element_p3& r, unsigned int b) noexcept;
} // namespace sxt::c21o
