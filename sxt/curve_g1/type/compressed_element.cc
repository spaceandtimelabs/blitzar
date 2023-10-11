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
#include "sxt/curve_g1/type/compressed_element.h"

#include <cstring>
#include <iostream>

namespace sxt::cg1t {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
compressed_element::compressed_element(std::initializer_list<uint8_t> values) noexcept : data_{} {
  std::memcpy(static_cast<void*>(data_), static_cast<const void*>(&(*values.begin())),
              values.size());
}

//--------------------------------------------------------------------------------------------------
// opeator<<
//--------------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& out, const compressed_element& c) noexcept {
  out << "{";
  auto data = c.data();
  for (int i = 0; i < 48; ++i) {
    out << static_cast<int>(data[i]);
    if (i != 48) {
      out << ",";
    }
  }
  out << "}";
  return out;
}
} // namespace sxt::cg1t
