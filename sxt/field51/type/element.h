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

namespace sxt::f51t {
//--------------------------------------------------------------------------------------------------
// element
//--------------------------------------------------------------------------------------------------
class element {
public:
  static constexpr size_t num_limbs_v = 5;

  element() noexcept = default;

  constexpr element(uint64_t x1, uint64_t x2, uint64_t x3, uint64_t x4, uint64_t x5) noexcept
      : data_{x1, x2, x3, x4, x5} {}

  constexpr const uint64_t& operator[](int index) const noexcept { return data_[index]; }

  constexpr uint64_t& operator[](int index) noexcept { return data_[index]; }

  constexpr const uint64_t* data() const noexcept { return data_; }

  constexpr uint64_t* data() noexcept { return data_; }

private:
  uint64_t data_[num_limbs_v];
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
} // namespace sxt::f51t
