#pragma once

#include <cstdint>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::algr {
//--------------------------------------------------------------------------------------------------
// test_add_reducer
//--------------------------------------------------------------------------------------------------
struct test_add_reducer {
  using value_type = uint64_t;

  template <class T> static inline CUDA_CALLABLE void accumulate(T& res, uint64_t x) noexcept {
    res = res + x;
  }

  template <class T>
  static inline CUDA_CALLABLE void accumulate_inplace(T& res, uint64_t x) noexcept {
    res = res + x;
  }
};
} // namespace sxt::algr
