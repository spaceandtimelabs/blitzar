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

namespace sxt::basn {
//--------------------------------------------------------------------------------------------------
// fast_random_number_generator
//--------------------------------------------------------------------------------------------------
class fast_random_number_generator {
public:
  CUDA_CALLABLE fast_random_number_generator(uint64_t seed1, uint64_t seed2) noexcept
      : state_{seed1, seed2} {}

  CUDA_CALLABLE
  uint64_t operator()() noexcept {
    // Uses the xorshift128p random number generation algorithm described in
    // https://en.wikipedia.org/wiki/Xorshift
    auto& state_a = state_[0];
    auto& state_b = state_[1];
    auto t = state_a;
    auto s = state_b;
    state_a = s;
    t ^= t << 23;       // a
    t ^= t >> 17;       // b
    t ^= s ^ (s >> 26); // c
    state_b = t;
    return t + s;
  }

private:
  uint64_t state_[2];
};
} // namespace sxt::basn
