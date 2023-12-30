#pragma once

#include <cmath>
#include <concepts>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::basn {
//--------------------------------------------------------------------------------------------------
// abs
//--------------------------------------------------------------------------------------------------
/**
 * Support abs for integral types larger than 128 bits.
 */
template <std::signed_integral T>
CUDA_CALLABLE T abs(T x) noexcept {
  if constexpr (sizeof(T) <= 8) {
    return std::abs(x);
  }

  // Note: There's probably a better way to do this that avoids branching, but
  // this is an ok place to start from.
  if (x < 0) {
    return -x;
  } else {
    return x;
  }
}
} // namespace sxt::basn
