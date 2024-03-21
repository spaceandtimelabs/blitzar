#pragma once

namespace sxt::basn {
//--------------------------------------------------------------------------------------------------
// choose_k 
//--------------------------------------------------------------------------------------------------
template <std::integral T>
constexpr T choose_k(T n, T k) noexcept {
  if (k == 0) {
    return 1;
  }
  T a = 1;
  T b = 1;
  for (unsigned j=0; j<k; ++j) {
    a *= n - j;
    b *= j + 1;
  }
  return a / b;
}
} // namespace sxt::basn
