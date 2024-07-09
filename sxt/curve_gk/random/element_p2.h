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
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/curve_gk/constant/generator.h"
#include "sxt/curve_gk/operation/scalar_multiply.h"
#include "sxt/curve_gk/type/element_p2.h"

namespace sxt::cgkrn {
//--------------------------------------------------------------------------------------------------
// generate_random_element
//--------------------------------------------------------------------------------------------------
/**
 * Generates a random scalar point and multiplies by
 * the generator to generate a random point on the curve.
 * Note: the random scalar does not respect the modulus of the curve's scalar field.
 * This is okay for benchmarks, but not for secure random element generation.
 */
CUDA_CALLABLE
inline void generate_random_element(cgkt::element_p2& a,
                                    basn::fast_random_number_generator& rng) noexcept {
  // Generate random scalar
  uint8_t data[32] = {};
  for (int i = 0; i < 32; i += 8) {
    auto x = rng();
    std::memcpy(static_cast<void*>(data + i), static_cast<void*>(&x), sizeof(x));
  }

  // Generate random point by multiplying the generator by the scalar
  cgko::scalar_multiply255(a, cgkcn::generator_p2_v, data);
}
} // namespace sxt::cgkrn
