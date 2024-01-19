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
struct element_p2;
}

namespace sxt::cg1o {
//--------------------------------------------------------------------------------------------------
// cmov
//--------------------------------------------------------------------------------------------------
/*
 Replace (f,g) with (g,g) if b == 1;
 replace (f,g) with (f,g) if b == 0.
 *
 Preconditions: b in {0,1}.
 */
CUDA_CALLABLE
void cmov(cg1t::element_p2& f, const cg1t::element_p2& g, unsigned int b) noexcept;
} // namespace sxt::cg1o
