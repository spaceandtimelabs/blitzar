#pragma once

#include <array>
#include <cstring>

#include "sxt/base/type/literal.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::s25t {
//--------------------------------------------------------------------------------------------------
// _s25
//--------------------------------------------------------------------------------------------------
template <char... Chars> element operator"" _s25() noexcept {
  std::array<uint64_t, 4> bytes = {};
  bast::parse_literal<Chars...>(bytes);
  element res;
  std::memcpy(static_cast<void*>(res.data()), static_cast<const void*>(bytes.data()), 32);
  return res;
}
} // namespace sxt::s25t
