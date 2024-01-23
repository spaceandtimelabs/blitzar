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

#include <cstdint>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::cg1t {
struct element_p2;
}

namespace sxt::cg1o {
//--------------------------------------------------------------------------------------------------
// scalar_multiply255
//--------------------------------------------------------------------------------------------------
/**
 * This is a simple double-and-add implementation of point multiplication, moving from most
 * significant to least significant bit of the scalar. We skip the leading bit because it's
 * always unset for Fq elements. Assumes the scalar q is little endian and the exponent has already
 * been reduced.
 */
CUDA_CALLABLE
void scalar_multiply255(cg1t::element_p2& h, const cg1t::element_p2& p,
                        const uint8_t q[32]) noexcept;
} // namespace sxt::cg1o
