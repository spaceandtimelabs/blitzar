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

#include "sxt/base/container/span.h"
#include "sxt/execution/async/future_fwd.h"

namespace sxt::c21t {
struct element_p3;
}

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// fold_generators_impl
//--------------------------------------------------------------------------------------------------
xena::future<void> fold_generators_impl(basct::span<c21t::element_p3> g_vector_p,
                                        basct::cspan<c21t::element_p3> g_vector,
                                        basct::cspan<unsigned> decomposition, size_t split_factor,
                                        size_t min_chunk_size, size_t max_chunk_size) noexcept;

//--------------------------------------------------------------------------------------------------
// fold_generators
//--------------------------------------------------------------------------------------------------
xena::future<void> fold_generators(basct::span<c21t::element_p3> g_vector_p,
                                   basct::cspan<c21t::element_p3> g_vector,
                                   basct::cspan<unsigned> decomposition) noexcept;
} // namespace sxt::prfip
