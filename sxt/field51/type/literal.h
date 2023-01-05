#pragma once

#include <array>

#include "sxt/base/type/literal.h"
#include "sxt/field51/base/byte_conversion.h"
#include "sxt/field51/type/element.h"

namespace sxt::f51t {
//--------------------------------------------------------------------------------------------------
// _f51
//--------------------------------------------------------------------------------------------------
template <char... Chars> element operator"" _f51() noexcept {
  std::array<uint64_t, 4> bytes = {};
  bast::parse_literal<4, Chars...>(bytes);
  element res;
  f51b::from_bytes(res.data(), reinterpret_cast<const uint8_t*>(bytes.data()));
  return res;
}
} // namespace sxt::f51t
