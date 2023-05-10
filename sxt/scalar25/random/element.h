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

namespace sxt::basn {
class fast_random_number_generator;
}
namespace sxt::s25t {
class element;
}

namespace sxt::s25rn {
//--------------------------------------------------------------------------------------------------
// generate_random_element
//--------------------------------------------------------------------------------------------------
void generate_random_element(s25t::element& e, basn::fast_random_number_generator& rng) noexcept;

//--------------------------------------------------------------------------------------------------
// generate_random_elements
//--------------------------------------------------------------------------------------------------
void generate_random_elements(basct::span<s25t::element> ex,
                              basn::fast_random_number_generator& rng) noexcept;
} // namespace sxt::s25rn
