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

#include <array>
#include <cstdint>
#include <iosfwd>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::f32t {
//--------------------------------------------------------------------------------------------------
// element
//--------------------------------------------------------------------------------------------------
class element {
public:
  static constexpr size_t num_limbs_v = 10;

  element() noexcept = default;

  CUDA_CALLABLE constexpr element(uint32_t x1, uint32_t x2, uint32_t x3, uint32_t x4, uint32_t x5,
                                  uint32_t x6, uint32_t x7, uint32_t x8, uint32_t x9,
                                  uint32_t x10) noexcept
      : data_{x1, x2, x3, x4, x5, x6, x7, x8, x9, x10} {}

  CUDA_CALLABLE constexpr const uint32_t& operator[](int index) const noexcept {
    return data_[index];
  }

  CUDA_CALLABLE constexpr uint32_t& operator[](int index) noexcept { return data_[index]; }

  CUDA_CALLABLE constexpr const uint32_t* data() const noexcept { return data_; }

  CUDA_CALLABLE constexpr uint32_t* data() noexcept { return data_; }

private:
  uint32_t data_[num_limbs_v];
};

//--------------------------------------------------------------------------------------------------
// operator<<
//--------------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& out, const element& e) noexcept;

//--------------------------------------------------------------------------------------------------
// operator==
//--------------------------------------------------------------------------------------------------
bool operator==(const element& lhs, const element& rhs) noexcept;

//--------------------------------------------------------------------------------------------------
// operator!=
//--------------------------------------------------------------------------------------------------
inline bool operator!=(const element& lhs, const element& rhs) noexcept { return !(lhs == rhs); }
} // namespace sxt::f32t
