#include "sxt/ristretto/type/compressed_element.h"

#include <cstring>
#include <iostream>

namespace sxt::rstt {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
compressed_element::compressed_element(std::initializer_list<uint8_t> values) noexcept
    : data_{} {
  std::memcpy(static_cast<void*>(data_),
              static_cast<const void*>(&(*values.begin())), values.size());
}

//--------------------------------------------------------------------------------------------------
// operator==
//--------------------------------------------------------------------------------------------------
bool operator==(const compressed_element& lhs, const compressed_element& rhs) noexcept {
  return std::memcmp(static_cast<const void*>(lhs.data()),
                     static_cast<const void*>(rhs.data()),
                     sizeof(compressed_element)) == 0;
}

//--------------------------------------------------------------------------------------------------
// opeator<<
//--------------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& out, const compressed_element& c) noexcept {
  out << "{";
  auto data = c.data();
  for (int i=0; i<32; ++i) {
    out << static_cast<int>(data[i]);
    if (i != 31) {
      out << ",";
    }
  }
  out << "}";
  return out;
}
} // namesapce sxt::rstt
