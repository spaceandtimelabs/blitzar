#include "sxt/multiexp/base/exponent.h"

#include <algorithm>
#include <tuple>

#include "sxt/base/bit/count.h"

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
exponent::exponent(uint64_t value1, uint64_t value2, uint64_t value3, uint64_t value4) noexcept
    : data_{value1, value2, value3, value4} {}

//--------------------------------------------------------------------------------------------------
// highest_bit
//--------------------------------------------------------------------------------------------------
int exponent::highest_bit() const noexcept {
  for (int i = 4; i-- > 0;) {
    if (data_[i] != 0) {
      static_assert(sizeof(unsigned long) == sizeof(uint64_t));
      return i * 64 + 63 - basbt::count_leading_zeros(data_[i]);
    }
  }
  return -1;
}

//--------------------------------------------------------------------------------------------------
// operator<
//--------------------------------------------------------------------------------------------------
bool exponent::operator<(const exponent& rhs) const noexcept {
  return std::make_tuple(data_[3], data_[2], data_[1], data_[0]) <
         std::make_tuple(rhs.data_[3], rhs.data_[2], rhs.data_[1], rhs.data_[0]);
}

//--------------------------------------------------------------------------------------------------
// operator==
//--------------------------------------------------------------------------------------------------
bool operator==(const exponent& lhs, const exponent& rhs) noexcept {
  return std::equal(lhs.data(), lhs.data() + 32, rhs.data());
}
} // namespace sxt::mtxb
