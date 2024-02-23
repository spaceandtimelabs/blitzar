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

#include <iosfwd>

#include "sxt/curve32/type/operation_adl_stub.h"
#include "sxt/field32/constant/one.h"
#include "sxt/field32/constant/zero.h"
#include "sxt/field32/type/element.h"

namespace sxt::c32t {
//--------------------------------------------------------------------------------------------------
// element_p3
//--------------------------------------------------------------------------------------------------
/**
 *  (extended): (X:Y:Z:T) satisfying x=X/Z, y=Y/Z, XY=ZT
 */
struct element_p3 : c32o::operation_adl_stub {
  element_p3() noexcept = default;

  constexpr element_p3(const f32t::element& X, const f32t::element& Y, const f32t::element& Z,
                       const f32t::element& T) noexcept
      : X{X}, Y{Y}, Z{Z}, T{T} {}

  f32t::element X;
  f32t::element Y;
  f32t::element Z;
  f32t::element T;

  static constexpr element_p3 identity() noexcept {
    return element_p3{f32cn::zero_v, f32cn::one_v, f32cn::one_v, f32cn::zero_v};
  }
};

//--------------------------------------------------------------------------------------------------
// mark
//--------------------------------------------------------------------------------------------------
void mark(element_p3& e) noexcept;

//--------------------------------------------------------------------------------------------------
// is_marked
//--------------------------------------------------------------------------------------------------
bool is_marked(const element_p3& e) noexcept;

//--------------------------------------------------------------------------------------------------
// operator==
//--------------------------------------------------------------------------------------------------
bool operator==(const element_p3& lhs, const element_p3& rhs) noexcept;

//--------------------------------------------------------------------------------------------------
// operator!=
//--------------------------------------------------------------------------------------------------
inline bool operator!=(const element_p3& lhs, const element_p3& rhs) noexcept {
  return !(lhs == rhs);
}

//--------------------------------------------------------------------------------------------------
// operator<<
//--------------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& out, const element_p3& e) noexcept;
} // namespace sxt::c32t
