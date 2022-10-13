#include "sxt/scalar25/type/element.h"

#include <cassert>
#include <cstring>
#include <iostream>

namespace sxt::s25t {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
element::element(std::initializer_list<uint8_t> values) noexcept : data_{} {
  assert(values.size() <= 32);
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
