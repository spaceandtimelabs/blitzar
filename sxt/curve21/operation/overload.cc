#include "sxt/curve21/operation/overload.h"

#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/scalar_multiply.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using sxt::s25t::operator""_s25;

namespace sxt::c21t {
//--------------------------------------------------------------------------------------------------
// operator+
//--------------------------------------------------------------------------------------------------
element_p3 operator+(const element_p3& lhs, const element_p3& rhs) noexcept {
  element_p3 res;
  c21o::add(res, lhs, rhs);
  return res;
}

//--------------------------------------------------------------------------------------------------
// operator-
//--------------------------------------------------------------------------------------------------
element_p3 operator-(const element_p3& lhs, const element_p3& rhs) noexcept {
  // Note: This isn't an efficient implementation, but that's ok as the overload operators
  // are really only used for making test code more readable.
  return lhs + -rhs;
}

element_p3 operator-(const element_p3& p) noexcept {
  // Note: This isn't an efficient implementation, but that's ok as the overload operators
  // are really only used for making test code more readable.
  return (-0x1_s25) * p;
}

//--------------------------------------------------------------------------------------------------
// operator*
//--------------------------------------------------------------------------------------------------
element_p3 operator*(const s25t::element& lhs, const element_p3& rhs) noexcept {
  element_p3 res;
  c21o::scalar_multiply(res, lhs, rhs);
  return res;
}

element_p3 operator*(uint64_t lhs, const element_p3& rhs) noexcept {
  element_p3 res;
  c21o::scalar_multiply(res, lhs, rhs);
  return res;
}

//--------------------------------------------------------------------------------------------------
// operator+=
//--------------------------------------------------------------------------------------------------
element_p3& operator+=(element_p3& lhs, const element_p3& rhs) noexcept {
  lhs = lhs + rhs;
  return lhs;
}

//--------------------------------------------------------------------------------------------------
// operator-=
//--------------------------------------------------------------------------------------------------
element_p3& operator-=(element_p3& lhs, const element_p3& rhs) noexcept {
  lhs = lhs - rhs;
  return lhs;
}
} // namespace sxt::c21t
