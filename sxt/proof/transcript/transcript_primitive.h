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

#include <type_traits>

namespace sxt::rstt {
struct compressed_element;
}

namespace sxt::s25t {
struct element;
}

namespace sxt::prft {
//--------------------------------------------------------------------------------------------------
// is_transcript_primitive_v
//--------------------------------------------------------------------------------------------------
template <class T>
constexpr bool is_transcript_primitive_v =
    std::is_integral_v<T> || std::is_same_v<T, unsigned char> ||
    std::is_same_v<T, rstt::compressed_element> || std::is_same_v<T, s25t::element>;
} // namespace sxt::prft
