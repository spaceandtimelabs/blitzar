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
#include "sxt/field12/type/element.h"

#include <array>
#include <iomanip>
#include <iostream>

#include "sxt/field12/base/byte_conversion.h"
#include "sxt/field12/base/reduce.h"

namespace sxt::f12t {
//--------------------------------------------------------------------------------------------------
// print_impl
//--------------------------------------------------------------------------------------------------
static std::ostream& print_impl(std::ostream& out, const std::array<uint8_t, 48>& bytes,
                                int start) noexcept {
  out << std::hex << static_cast<int>(bytes[start]);
  for (int i = start; i-- > 0;) {
    out << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(bytes[i]);
  }
  out << "_f12";
  return out;
}

//--------------------------------------------------------------------------------------------------
// operator<<
//--------------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& out, const element& e) noexcept {
  std::array<uint8_t, 48> bytes = {};
  f12b::to_bytes(bytes.data(), e.data());
  auto flags = out.flags();
  out << "0x";
  for (int i = 48; i-- > 0;) {
    if (bytes[i] != 0) {
      print_impl(out, bytes, i);
      out.flags(flags);
      return out;
    }
  }
  out << "0_f12";
  out.flags(flags);
  return out;
}

//--------------------------------------------------------------------------------------------------
// operator==
//--------------------------------------------------------------------------------------------------
bool operator==(const element& lhs, const element& rhs) noexcept {
  element lhs_p;
  f12b::reduce(lhs_p.data(), lhs.data());

  element rhs_p;
  f12b::reduce(rhs_p.data(), rhs.data());

  return lhs_p[0] == rhs_p[0] && lhs_p[1] == rhs_p[1] && lhs_p[2] == rhs_p[2] &&
         lhs_p[3] == rhs_p[3] && lhs_p[4] == rhs_p[4] && lhs_p[5] == rhs_p[5];
}
} // namespace sxt::f12t
