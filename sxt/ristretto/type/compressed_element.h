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

#include <cstdint>
#include <initializer_list>
#include <iosfwd>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::rstt {
//--------------------------------------------------------------------------------------------------
// compressed_element
//--------------------------------------------------------------------------------------------------
class compressed_element {
public:
  compressed_element() noexcept = default;

  explicit compressed_element(std::initializer_list<uint8_t> values) noexcept;

  CUDA_CALLABLE
  uint8_t* data() noexcept { return data_; }

  CUDA_CALLABLE
  const uint8_t* data() const noexcept { return data_; }

private:
  uint8_t data_[32] = {};
};

//--------------------------------------------------------------------------------------------------
// operator==
//--------------------------------------------------------------------------------------------------
bool operator==(const compressed_element& lhs, const compressed_element& rhs) noexcept;

//--------------------------------------------------------------------------------------------------
// operator!=
//--------------------------------------------------------------------------------------------------
inline bool operator!=(const compressed_element& lhs, const compressed_element& rhs) noexcept {
  return !(lhs == rhs);
}

//--------------------------------------------------------------------------------------------------
// opeator<<
//--------------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& out, const compressed_element& c) noexcept;
} // namespace sxt::rstt
