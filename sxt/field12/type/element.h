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

namespace sxt::f12t {
//--------------------------------------------------------------------------------------------------
// element
//--------------------------------------------------------------------------------------------------
class element {
public:
  element() noexcept = default;

  CUDA_CALLABLE constexpr element(uint64_t x1, uint64_t x2, uint64_t x3, uint64_t x4, uint64_t x5,
                                  uint64_t x6) noexcept
      : data_{x1, x2, x3, x4, x5, x6} {}

  CUDA_CALLABLE constexpr element(const uint64_t x[6]) noexcept
      : data_{x[0], x[1], x[2], x[3], x[4], x[5]} {}

  constexpr element(const element&) noexcept = default;

  CUDA_CALLABLE element(const volatile element& other) noexcept {
    for (int i = 0; i < 6; ++i) {
      data_[i] = other.data_[i];
    }
  }

  constexpr element& operator=(const element&) noexcept = default;

  constexpr element& operator=(element&&) noexcept = default;

  CUDA_CALLABLE constexpr const uint64_t& operator[](int index) const noexcept {
    return data_[index];
  }

  CUDA_CALLABLE constexpr const volatile uint64_t& operator[](int index) const volatile noexcept {
    return data_[index];
  }

  CUDA_CALLABLE constexpr uint64_t& operator[](int index) noexcept { return data_[index]; }

  CUDA_CALLABLE constexpr volatile uint64_t& operator[](int index) volatile noexcept {
    return data_[index];
  }

  CUDA_CALLABLE constexpr const uint64_t* data() const noexcept { return data_; }

  CUDA_CALLABLE constexpr uint64_t* data() noexcept { return data_; }

private:
  uint64_t data_[6];
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
} // namespace sxt::f12t
