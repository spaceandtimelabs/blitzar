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
#include "sxt/field32/type/element.h"

#include <array>
#include <iomanip>
#include <iostream>

#include "sxt/field32/base/byte_conversion.h"
#include "sxt/field32/base/reduce.h"

namespace sxt::f32t {
//--------------------------------------------------------------------------------------------------
// print_impl
//--------------------------------------------------------------------------------------------------
static std::ostream& print_impl(std::ostream& out, const std::array<uint8_t, 32>& bytes,
                                int start) noexcept {
  out << std::hex << static_cast<int>(bytes[start]);
  for (int i = start; i-- > 0;) {
    out << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(bytes[i]);
  }
  out << "_f32";
  return out;
}

//--------------------------------------------------------------------------------------------------
// operator<<
//--------------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& out, const element& e) noexcept {
  std::array<uint8_t, 32> bytes = {};
  f32b::to_bytes(bytes.data(), e.data());
  auto flags = out.flags();
  out << "0x";
  for (int i = 32; i-- > 0;) {
    if (bytes[i] != 0) {
      print_impl(out, bytes, i);
      out.flags(flags);
      return out;
    }
  }
  out << "0_f32";
  out.flags(flags);
  return out;
}

//--------------------------------------------------------------------------------------------------
// operator==
//--------------------------------------------------------------------------------------------------
/*
 * Might not be working. Need to test.
 */
bool operator==(const element& lhs, const element& rhs) noexcept {
  element lhs_p;
  uint64_t lhs_data[10] = {lhs[0], lhs[1], lhs[2], lhs[3], lhs[4],
                           lhs[5], lhs[6], lhs[7], lhs[8], lhs[9]};
  f32b::reduce(lhs_p.data(), lhs_data);

  element rhs_p;
  uint64_t rhs_data[10] = {rhs[0], rhs[1], rhs[2], rhs[3], rhs[4],
                           rhs[5], rhs[6], rhs[7], rhs[8], rhs[9]};
  f32b::reduce(rhs_p.data(), rhs_data);

  return std::equal(lhs_p.data(), lhs_p.data() + element::num_limbs_v, rhs_p.data());
}
} // namespace sxt::f32t
