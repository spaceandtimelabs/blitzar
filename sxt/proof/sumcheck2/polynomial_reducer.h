#pragma once

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/field/element.h"

namespace sxt::prfsk2 {
//--------------------------------------------------------------------------------------------------
// polynomial_reducer
//--------------------------------------------------------------------------------------------------
template <unsigned Degree, basfld::element T> struct polynomial_reducer {
  using value_type = std::array<T, Degree + 1u>;

  CUDA_CALLABLE static void accumulate_inplace(value_type& res, const value_type& e) noexcept {
    for (unsigned i = 0; i < res.size(); ++i) {
      add(res[i], res[i], e[i]);
    }
  }
};
} // namespace sxt::prfsk2
