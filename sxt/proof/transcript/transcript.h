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
#include <string>

#include "sxt/base/container/span.h"
#include "sxt/proof/transcript/strobe128.h"

namespace sxt::prft {
//--------------------------------------------------------------------------------------------------
// transcript
//--------------------------------------------------------------------------------------------------
class transcript {
public:
  explicit transcript(std::string_view label) noexcept;

  void append_message(std::string_view label, basct::cspan<uint8_t> message) noexcept;

  void challenge_bytes(basct::span<uint8_t> dest, std::string_view label) noexcept;

private:
  strobe128 strobe_;
};

//--------------------------------------------------------------------------------------------------
// operator==
//--------------------------------------------------------------------------------------------------
bool operator==(const transcript& lhs, const transcript& rhs) noexcept;

//--------------------------------------------------------------------------------------------------
// operator!=
//--------------------------------------------------------------------------------------------------
inline bool operator!=(const transcript& lhs, const transcript& rhs) noexcept {
  return !(lhs == rhs);
}
} // namespace sxt::prft
