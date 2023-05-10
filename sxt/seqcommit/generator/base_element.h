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

namespace sxt::c21t {
struct element_p3;
}
namespace sxt::rstt {
struct compressed_element;
}

namespace sxt::sqcgn {
//--------------------------------------------------------------------------------------------------
// compute_base_element
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void compute_base_element(c21t::element_p3& g, uint64_t index) noexcept;

//--------------------------------------------------------------------------------------------------
// compute_compressed_base_element
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void compute_compressed_base_element(rstt::compressed_element& g_rt, uint64_t index) noexcept;
} // namespace sxt::sqcgn
