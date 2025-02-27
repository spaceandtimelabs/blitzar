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
#include "sxt/proof/transcript/transcript_utility.h"

#include "sxt/fieldgk/base/byte_conversion.h"
#include "sxt/fieldgk/type/element.h"
#include "sxt/scalar25/operation/reduce.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prft {
//--------------------------------------------------------------------------------------------------
// challenge_value
//--------------------------------------------------------------------------------------------------
void challenge_value(s25t::element& value, transcript& trans, std::string_view label) noexcept {
  trans.challenge_bytes({reinterpret_cast<uint8_t*>(&value), sizeof(s25t::element)}, label);
  s25o::reduce32(value);
}

void challenge_value(fgkt::element& value, transcript& trans, std::string_view label) noexcept {
  uint64_t data[4];
  trans.challenge_bytes({reinterpret_cast<uint8_t*>(data), 32}, label);
  fgkb::to_bytes_le(reinterpret_cast<uint8_t*>(value.data()), data);
}

//--------------------------------------------------------------------------------------------------
// challenge_values
//--------------------------------------------------------------------------------------------------
void challenge_values(basct::span<s25t::element> values, transcript& trans,
                      std::string_view label) noexcept {
  trans.challenge_bytes(
      {reinterpret_cast<uint8_t*>(values.data()), values.size() * sizeof(s25t::element)}, label);
  for (auto& val : values) {
    s25o::reduce32(val);
  }
}
} // namespace sxt::prft
