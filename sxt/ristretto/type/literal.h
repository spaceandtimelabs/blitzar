#pragma once

#include "sxt/base/type/literal.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/field51/base/byte_conversion.h"
#include "sxt/field51/type/element.h"
#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/base/point_formation.h"
#include "sxt/ristretto/type/compressed_element.h"

namespace sxt::rstt {
//--------------------------------------------------------------------------------------------------
// _rs
//--------------------------------------------------------------------------------------------------
template <char... Chars> c21t::element_p3 operator"" _rs() noexcept {
  std::array<uint64_t, 8> bytes;
  bast::parse_literal<8, Chars...>(bytes);
  f51t::element x, y;
  f51b::from_bytes(x.data(), reinterpret_cast<const uint8_t*>(bytes.data()));
  f51b::from_bytes(y.data(), reinterpret_cast<const uint8_t*>(&bytes[4]));
  c21t::element_p3 res;
  rstb::form_ristretto_point(res, x, y);
  return res;
}

//--------------------------------------------------------------------------------------------------
// _crs
//--------------------------------------------------------------------------------------------------
template <char... Chars> compressed_element operator"" _crs() noexcept {
  auto e = operator""_rs<Chars...>();
  compressed_element res;
  rstb::to_bytes(res.data(), e);
  return res;
}
} // namespace sxt::rstt
