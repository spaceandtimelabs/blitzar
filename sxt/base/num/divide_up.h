#pragma once

#include <type_traits>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::basn {
//--------------------------------------------------------------------------------------------------
// divide_up
//--------------------------------------------------------------------------------------------------
template <class T>
  requires std::is_integral_v<T>
CUDA_CALLABLE inline T divide_up(T a, T b) noexcept {
  return (a + b - 1) / b;
}
} // namespace sxt::basn
