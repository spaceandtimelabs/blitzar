#pragma once

#include <cstdint>

#include "sxt/base/type/literal.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/curve21/type/point_formation.h"

namespace sxt::c21t {
//--------------------------------------------------------------------------------------------------
// operator _c21
//--------------------------------------------------------------------------------------------------
template <char... Chars> element_p3 operator"" _c21() noexcept {
  std::array<uint64_t, 4> bytes = {};
  bast::parse_literal<4, Chars...>(bytes);
  element_p3 res;
  form_point(res, reinterpret_cast<const uint8_t*>(bytes.data()));
  return res;
}
} // namespace sxt::c21t
