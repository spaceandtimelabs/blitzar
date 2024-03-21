#pragma once

#include <concepts>

#include "sxt/base/bit/count.h"

namespace sxt::basbt {
//--------------------------------------------------------------------------------------------------
// next_permutation 
//--------------------------------------------------------------------------------------------------
// adopted from https://stackoverflow.com/a/8281965
template <std::unsigned_integral T>
T next_permutation(T x) noexcept {
  static constexpr T one{1};
  T t = x | (x - one);
  return (t + one) | (((~t & -~t) - one) >> (basbt::count_trailing_zeros(x) + 1));
}
} // namespace sxt::basbt
