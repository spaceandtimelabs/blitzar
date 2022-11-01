#pragma once

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::basn {
//--------------------------------------------------------------------------------------------------
// is_power2
//--------------------------------------------------------------------------------------------------
// returns 1 in case n is a power of 2. Returns 0 otherwise
CUDA_CALLABLE
inline bool is_power2(unsigned long long n) noexcept { return (n != 0) && ((n & (n - 1)) == 0); }
} // namespace sxt::basn
