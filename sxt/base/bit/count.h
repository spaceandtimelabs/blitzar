#pragma once

namespace sxt::basbt {
//--------------------------------------------------------------------------------------------------
// count_trailing_zeros
//--------------------------------------------------------------------------------------------------
inline int count_trailing_zeros(unsigned long x) noexcept {
  return __builtin_ctzl(x);
}
} // namespace sxt::basbt
