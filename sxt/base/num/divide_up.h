#pragma once

#include <type_traits>

namespace sxt::basn {
//--------------------------------------------------------------------------------------------------
// divide_up
//--------------------------------------------------------------------------------------------------
template <class T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
inline T divide_up(T a, T b) noexcept {
  return (a + b - 1) / b;
}
} // namespace sxt::basn
