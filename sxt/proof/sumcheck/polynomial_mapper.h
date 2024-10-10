#pragma once

#include <array>
#include <utility>

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/scalar25/operation/add.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// polynomial_mapper
//--------------------------------------------------------------------------------------------------
template <unsigned MaxDegree> struct polynomial_mapper {
  using value_type = std::array<s25t::element, MaxDegree + 1u>;

  CUDA_CALLABLE
  value_type map_index(unsigned index) const noexcept {
    (void)index;
    return {};
  }

  CUDA_CALLABLE
  void map_index(value_type& p, unsigned index) const noexcept {
    (void)p;
    (void)index;
  }
};
} // namespace sxt::prfsk
