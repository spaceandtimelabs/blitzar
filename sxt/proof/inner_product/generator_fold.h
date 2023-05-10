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

namespace sxt::c21t {
struct element_p3;
}
namespace sxt::s25t {
class element;
}

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// decompose_generator_fold
//--------------------------------------------------------------------------------------------------
void decompose_generator_fold(basct::span<unsigned>& res, const s25t::element& m_low,
                              const s25t::element& m_high) noexcept;

//--------------------------------------------------------------------------------------------------
// fold_generators
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE void fold_generators(c21t::element_p3& res, basct::cspan<unsigned> decomposition,
                                   const c21t::element_p3& g_low,
                                   const c21t::element_p3& g_high) noexcept;
} // namespace sxt::prfip
