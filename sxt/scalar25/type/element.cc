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
#include "sxt/scalar25/type/element.h"

#include <cstring>
#include <iostream>

#include "sxt/base/error/assert.h"

namespace sxt::s25t {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
element::element(std::initializer_list<uint8_t> values) noexcept : data_{} {
  SXT_DEBUG_ASSERT(values.size() <= 32);
  std::memcpy(static_cast<void*>(data_), static_cast<const void*>(&(*values.begin())),
              values.size());
}

//--------------------------------------------------------------------------------------------------
// operator==
//--------------------------------------------------------------------------------------------------
bool operator==(const element& lhs, const element& rhs) noexcept {
  return std::memcmp(static_cast<const void*>(lhs.data()), static_cast<const void*>(rhs.data()),
                     sizeof(element)) == 0;
}

//--------------------------------------------------------------------------------------------------
// operator<<
//--------------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& out, const element& c) noexcept {
  out << "{";
  auto data = c.data();
  for (int i = 0; i < 32; ++i) {
    out << static_cast<int>(data[i]);
    if (i != 31) {
      out << ",";
    }
  }
  out << "}";
  return out;
}
} // namespace sxt::s25t
