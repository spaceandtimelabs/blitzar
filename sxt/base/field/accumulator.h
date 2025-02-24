#pragma once

#include "sxt/base/field/element.h"
#include "sxt/base/macro/cuda_callable.h"

namespace sxt::basfld {
//--------------------------------------------------------------------------------------------------
// accumulator
//--------------------------------------------------------------------------------------------------
template <basfld::element T>
struct accumulator {
  using value_type = T;

  CUDA_CALLABLE static void accumulate_inplace(T& res, T& e) noexcept {
    add(res, res, e);
  }
};
} // namespace sxt::basfld
