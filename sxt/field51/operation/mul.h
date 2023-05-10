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

#include <cstdint>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::f51t {
class element;
}

namespace sxt::f51o {
//--------------------------------------------------------------------------------------------------
// mul
//--------------------------------------------------------------------------------------------------
/*
 h = f * g
 Can overlap h with f or g.
 */
CUDA_CALLABLE
void mul(f51t::element& h, const f51t::element& f, const f51t::element& g) noexcept;

CUDA_CALLABLE
void mul(volatile f51t::element& h, const f51t::element& f, const f51t::element& g) noexcept;

CUDA_CALLABLE
void mul(volatile f51t::element& h, const volatile f51t::element& f,
         const volatile f51t::element& g) noexcept;

CUDA_CALLABLE
void mul(volatile f51t::element& h, const volatile f51t::element& f,
         const f51t::element& g) noexcept;

//--------------------------------------------------------------------------------------------------
// mul32
//--------------------------------------------------------------------------------------------------
void mul32(f51t::element& h, const f51t::element& f, uint32_t n) noexcept;
} // namespace sxt::f51o
