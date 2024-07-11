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

#include <cstring>

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/curve_bng1/random/element_p2.h"
#include "sxt/curve_bng1/type/conversion_utility.h"
#include "sxt/curve_bng1/type/element_affine.h"
#include "sxt/curve_bng1/type/element_p2.h"

namespace sxt::cn1rn {
//--------------------------------------------------------------------------------------------------
// generate_random_element
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
inline void generate_random_element(cn1t::element_affine& a,
                                    basn::fast_random_number_generator& rng) noexcept {
  cn1t::element_p2 p;
  generate_random_element(p, rng);
  cn1t::to_element_affine(a, p);
}
} // namespace sxt::cn1rn
