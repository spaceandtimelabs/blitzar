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

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::basfld {
//--------------------------------------------------------------------------------------------------
// element4
//--------------------------------------------------------------------------------------------------
/**
 * Provides a minimal implementation of the field element concent that can be used for writing
 * tests.
 *
 * element4 uses the limb count and modulus (in little endian ordering) of the bn254
 * curve.
 */
struct element4 {
  static constexpr size_t num_limbs_v = 4;

  element4() noexcept = default;

  CUDA_CALLABLE constexpr element4(uint64_t x1, uint64_t x2, uint64_t x3, uint64_t x4) noexcept
      : data_{x1, x2, x3, x4} {}

  CUDA_CALLABLE constexpr const uint64_t& operator[](int index) const noexcept {
    return data_[index];
  }

  CUDA_CALLABLE constexpr uint64_t& operator[](int index) noexcept { return data_[index]; }

  CUDA_CALLABLE constexpr const uint64_t* data() const noexcept { return data_; }

  CUDA_CALLABLE constexpr uint64_t* data() noexcept { return data_; }

  static constexpr element4 modulus() noexcept {
    return element4{0x3c208c16d87cfd47, 0x97816a916871ca8d, 0xb85045b68181585d, 0x30644e72e131a029};
  }

  bool operator==(const element4&) const noexcept = default;

private:
  uint64_t data_[num_limbs_v];
};
} // namespace sxt::basfld
