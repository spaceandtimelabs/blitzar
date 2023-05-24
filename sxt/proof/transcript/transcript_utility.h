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

#include <string>
#include <type_traits>

#include "sxt/base/container/span.h"
#include "sxt/proof/transcript/transcript.h"
#include "sxt/proof/transcript/transcript_primitive.h"

namespace sxt::s25t {
class element;
}

namespace sxt::prft {
//--------------------------------------------------------------------------------------------------
// append_value
//--------------------------------------------------------------------------------------------------
template <class T, std::enable_if_t<is_transcript_primitive_v<T>>* = nullptr>
inline void append_value(transcript& trans, std::string_view label, const T& value) noexcept {
  trans.append_message(label, {reinterpret_cast<const uint8_t*>(&value), sizeof(T)});
}

inline void append_value(transcript& trans, std::string_view label,
                         std::string_view value) noexcept {
  trans.append_message(label, {reinterpret_cast<const uint8_t*>(value.data()), value.size()});
}

//--------------------------------------------------------------------------------------------------
// append_values
//--------------------------------------------------------------------------------------------------
template <class T, std::enable_if_t<is_transcript_primitive_v<T>>* = nullptr>
inline void append_values(transcript& trans, std::string_view label,
                          basct::cspan<T> values) noexcept {
  trans.append_message(
      label, {reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(T)});
}

//--------------------------------------------------------------------------------------------------
// challenge_value
//--------------------------------------------------------------------------------------------------
void challenge_value(s25t::element& value, transcript& trans, std::string_view label) noexcept;

//--------------------------------------------------------------------------------------------------
// challenge_values
//--------------------------------------------------------------------------------------------------
void challenge_values(basct::span<s25t::element> values, transcript& trans,
                      std::string_view label) noexcept;

//--------------------------------------------------------------------------------------------------
// set_domain
//--------------------------------------------------------------------------------------------------
inline void set_domain(transcript& trans, std::string_view domain_name) noexcept {
  append_value(trans, "domain-sep", domain_name);
}
} // namespace sxt::prft
