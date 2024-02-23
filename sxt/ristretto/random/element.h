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

#include <random>

#include "sxt/base/container/span.h"
#include "sxt/base/macro/cuda_callable.h"

namespace sxt::basn {
class fast_random_number_generator;
}
namespace sxt::c32t {
struct element_p3;
}
namespace sxt::rstt {
class compressed_element;
}

namespace sxt::rstrn {
//--------------------------------------------------------------------------------------------------
// generate_random_element
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void generate_random_element(c32t::element_p3& p, basn::fast_random_number_generator& rng) noexcept;

void generate_random_element(c32t::element_p3& p, std::mt19937& rng) noexcept;

void generate_random_element(rstt::compressed_element& p,
                             basn::fast_random_number_generator& rng) noexcept;

//--------------------------------------------------------------------------------------------------
// generate_random_elements
//--------------------------------------------------------------------------------------------------
void generate_random_elements(basct::span<c32t::element_p3> px,
                              basn::fast_random_number_generator& rng) noexcept;

void generate_random_elements(basct::span<c32t::element_p3> px, std::mt19937& rng) noexcept;

void generate_random_elements(basct::span<rstt::compressed_element> px,
                              basn::fast_random_number_generator& rng) noexcept;
} // namespace sxt::rstrn
