#pragma once

#include <array>
#include <utility>

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/scalar25/operation/add.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// polynomial_reducer
//--------------------------------------------------------------------------------------------------
template <unsigned MaxDegree> struct polynomial_reducer {
  using value_type = std::array<s25t::element, MaxDegree + 1u>;

  CUDA_CALLABLE static void accumulate_inplace(value_type& res, const value_type& e) noexcept {
    for (unsigned i=0; i<MaxDegree+1u; ++i) {
      s25o::add(res[i], res[i], e[i]);
    }
  }
};
} // namespace sxt::prfsk
