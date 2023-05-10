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
#include "sxt/curve21/type/element_p3.h"

#include <iostream>

#include "sxt/field51/operation/mul.h"

namespace sxt::c21t {
//--------------------------------------------------------------------------------------------------
// operator==
//--------------------------------------------------------------------------------------------------
bool operator==(const element_p3& lhs, const element_p3& rhs) noexcept {
  f51t::element lhs_p, rhs_p;
  f51o::mul(lhs_p, lhs.X, rhs.Z);
  f51o::mul(rhs_p, rhs.X, lhs.Z);
  if (lhs_p != rhs_p) {
    return false;
  }
  f51o::mul(lhs_p, lhs.Y, rhs.Z);
  f51o::mul(rhs_p, rhs.Y, lhs.Z);
  return lhs_p == rhs_p;
}

//--------------------------------------------------------------------------------------------------
// operator<<
//--------------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& out, const element_p3& e) noexcept {
  out << "{ .X=" << e.X << ", .Y=" << e.Y << ", .Z=" << e.Z << ", .T=" << e.T << "}";
  return out;
}
} // namespace sxt::c21t
