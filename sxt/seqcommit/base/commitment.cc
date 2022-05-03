#include "sxt/seqcommit/base/commitment.h"

#include <cstring>
#include <iostream>

namespace sxt::sqcb {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
commitment::commitment(std::initializer_list<uint8_t> values) noexcept
    : data_{} {
  std::memcpy(static_cast<void*>(data_),
              static_cast<const void*>(&(*values.begin())), values.size());
}

//--------------------------------------------------------------------------------------------------
// operator==
//--------------------------------------------------------------------------------------------------
bool operator==(const commitment& lhs, const commitment& rhs) noexcept {
  return std::memcmp(static_cast<const void*>(lhs.data()),
                     static_cast<const void*>(rhs.data()),
                     sizeof(commitment)) == 0;
}

//--------------------------------------------------------------------------------------------------
// opeator<<
//--------------------------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& out, const commitment& c) noexcept {
  out << "{";
  auto data = c.data();
  for (int i=0; i<32; ++i) {
    out << static_cast<int>(data[i]);
    if (i != 31) {
      out << ",";
    }
  }
  out << "}";
  (void)c;
  return out;
}
} // namespace sxt::sqcb
