#include "sxt/ristretto/operation/overload.h"

#include "sxt/curve21/operation/overload.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/ristretto/operation/compression.h"
#include "sxt/ristretto/type/compressed_element.h"

namespace sxt::s25t {
class element;
}

namespace sxt::rstt {
//--------------------------------------------------------------------------------------------------
// operator+
//--------------------------------------------------------------------------------------------------
compressed_element operator+(const compressed_element& lhs,
                             const compressed_element& rhs) noexcept {
  c21t::element_p3 lhs_p, rhs_p;
  rsto::decompress(lhs_p, lhs);
  rsto::decompress(rhs_p, rhs);
  auto res_p = lhs_p + rhs_p;
  compressed_element res;
  rsto::compress(res, res_p);
  return res;
}

compressed_element operator+(const c21t::element_p3& lhs, const compressed_element& rhs) noexcept {
  compressed_element lhs_p;
  rsto::compress(lhs_p, lhs);
  return lhs_p + rhs;
}

compressed_element operator+(const compressed_element& lhs, const c21t::element_p3& rhs) noexcept {
  return rhs + lhs;
}

//--------------------------------------------------------------------------------------------------
// operator-
//--------------------------------------------------------------------------------------------------
compressed_element operator-(const compressed_element& lhs,
                             const compressed_element& rhs) noexcept {
  c21t::element_p3 lhs_p, rhs_p;
  rsto::decompress(lhs_p, lhs);
  rsto::decompress(rhs_p, rhs);
  auto res_p = lhs_p - rhs_p;
  compressed_element res;
  rsto::compress(res, res_p);
  return res;
}

compressed_element operator-(const compressed_element& x) noexcept {
  c21t::element_p3 x_p;
  rsto::decompress(x_p, x);
  auto res_p = -x_p;
  compressed_element res;
  rsto::compress(res, res_p);
  return res;
}

compressed_element operator-(const c21t::element_p3& lhs, const compressed_element& rhs) noexcept {
  compressed_element lhs_p;
  rsto::compress(lhs_p, lhs);
  return lhs_p - rhs;
}

compressed_element operator-(const compressed_element& lhs, const c21t::element_p3& rhs) noexcept {
  compressed_element rhs_p;
  rsto::compress(rhs_p, rhs);
  return lhs - rhs_p;
}

//--------------------------------------------------------------------------------------------------
// operator*
//--------------------------------------------------------------------------------------------------
compressed_element operator*(const s25t::element& lhs, const compressed_element& rhs) noexcept {
  c21t::element_p3 rhs_p;
  rsto::decompress(rhs_p, rhs);
  auto res_p = lhs * rhs_p;
  compressed_element res;
  rsto::compress(res, res_p);
  return res;
}

//--------------------------------------------------------------------------------------------------
// operator+=
//--------------------------------------------------------------------------------------------------
compressed_element& operator+=(compressed_element& lhs, const compressed_element& rhs) noexcept {
  c21t::element_p3 lhs_p, rhs_p;
  rsto::decompress(lhs_p, lhs);
  rsto::decompress(rhs_p, rhs);
  lhs_p += rhs_p;
  rsto::compress(lhs, lhs_p);
  return lhs;
}

//--------------------------------------------------------------------------------------------------
// operator-=
//--------------------------------------------------------------------------------------------------
compressed_element& operator-=(compressed_element& lhs, const compressed_element& rhs) noexcept {
  c21t::element_p3 lhs_p, rhs_p;
  rsto::decompress(lhs_p, lhs);
  rsto::decompress(rhs_p, rhs);
  lhs_p -= rhs_p;
  rsto::compress(lhs, lhs_p);
  return lhs;
}
} // namespace sxt::rstt
