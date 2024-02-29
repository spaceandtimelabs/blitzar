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

namespace sxt::cg1t {
struct element_affine;
struct element_p2;
} // namespace sxt::cg1t

namespace sxt::cg1p {
//--------------------------------------------------------------------------------------------------
// is_on_curve
//--------------------------------------------------------------------------------------------------
bool is_on_curve(const cg1t::element_affine& p) noexcept;

//--------------------------------------------------------------------------------------------------
// is_on_curve
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
bool is_on_curve(const cg1t::element_p2& p) noexcept;
} // namespace sxt::cg1p
