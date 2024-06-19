#pragma once

#include <concepts>

namespace sxt::basn {
//--------------------------------------------------------------------------------------------------
// round_up
//--------------------------------------------------------------------------------------------------
template <class T>
  requires std::is_integral_v<T>
constexpr inline T round_up(T a, T multiple) noexcept {
  auto t = (a + multiple - 1) / multiple;
  return t * multiple;
}
} // namespace sxt::basn
