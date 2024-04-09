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

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::basn {
class fast_random_number_generator;
}
namespace sxt::f32t {
class element;
}

namespace sxt::f32rn {
//--------------------------------------------------------------------------------------------------
// generate_random_element
//--------------------------------------------------------------------------------------------------
/**
 * Not guaranteed to be uniform. Only random elements generated below the modulus will be accepted.
 */
CUDA_CALLABLE
void generate_random_element(f32t::element& e, basn::fast_random_number_generator& rng) noexcept;

void generate_random_element(f32t::element& e, std::mt19937& rng) noexcept;
} // namespace sxt::f32rn
