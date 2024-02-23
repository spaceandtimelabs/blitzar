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

namespace sxt::c32t {
struct element_p3;
}
namespace sxt::s25t {
class element;
}

namespace sxt::rstt {
class compressed_element;

//--------------------------------------------------------------------------------------------------
// operator+
//--------------------------------------------------------------------------------------------------
compressed_element operator+(const compressed_element& lhs, const compressed_element& rhs) noexcept;
compressed_element operator+(const c32t::element_p3& lhs, const compressed_element& rhs) noexcept;
compressed_element operator+(const compressed_element& lhs, const c32t::element_p3& rhs) noexcept;

//--------------------------------------------------------------------------------------------------
// operator-
//--------------------------------------------------------------------------------------------------
compressed_element operator-(const compressed_element& lhs, const compressed_element& rhs) noexcept;
compressed_element operator-(const compressed_element& x) noexcept;
compressed_element operator-(const c32t::element_p3& lhs, const compressed_element& rhs) noexcept;
compressed_element operator-(const compressed_element& lhs, const c32t::element_p3& rhs) noexcept;

//--------------------------------------------------------------------------------------------------
// operator*
//--------------------------------------------------------------------------------------------------
compressed_element operator*(const s25t::element& lhs, const compressed_element& rhs) noexcept;

//--------------------------------------------------------------------------------------------------
// operator+=
//--------------------------------------------------------------------------------------------------
compressed_element& operator+=(compressed_element& lhs, const compressed_element& rhs) noexcept;

//--------------------------------------------------------------------------------------------------
// operator-=
//--------------------------------------------------------------------------------------------------
compressed_element& operator-=(compressed_element& lhs, const compressed_element& rhs) noexcept;
} // namespace sxt::rstt
