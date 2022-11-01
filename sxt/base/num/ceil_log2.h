#pragma once

#include <cassert>
#include <cstdint>

#include "sxt/base/bit/count.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/num/power2_equality.h"

namespace sxt::basn {
//--------------------------------------------------------------------------------------------------
// ceil_log2
//--------------------------------------------------------------------------------------------------
// returns ceil(log2(n)).
//
// assert in case n is zero
CUDA_CALLABLE
inline int ceil_log2(uint64_t n) noexcept {
  assert(n != 0);

  if (is_power2(n)) {
    return basbt::count_trailing_zeros(n);
  }

  return 64 - basbt::count_leading_zeros(n);
}
} // namespace sxt::basn
