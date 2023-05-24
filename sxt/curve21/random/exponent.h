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

namespace sxt::c21rn {
//--------------------------------------------------------------------------------------------------
// generate_random_exponent
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
inline void generate_random_exponent(unsigned char a[32],
                                     basn::fast_random_number_generator& generator) noexcept {
  for (int i = 0; i < 32; i += 8) {
    auto x = generator();
    std::memcpy(static_cast<void*>(a + i), static_cast<void*>(&x), sizeof(x));
  }
  // make sure a[31] <= 127
  a[31] = a[31] >> 1;
}
} // namespace sxt::c21rn
