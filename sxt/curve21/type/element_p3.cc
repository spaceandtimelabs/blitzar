#include "sxt/curve21/type/element_p3.h"

#include <iostream>

#include "sxt/field51/operation/mul.h"

#include <iostream>

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
  out << "{ .X=" << e.X << ", .Y=" << e.Y << ", .Z=" << e.Z << ", .T=" << e.T
      << "}";
  return out;
}
} // namespace sxt::c21t
