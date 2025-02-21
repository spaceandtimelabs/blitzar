#pragma once

#include "sxt/base/field/element.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/proof/sumcheck2/polynomial_utility.h"

namespace sxt::prfsk2 {
//--------------------------------------------------------------------------------------------------
// polynomial_mapper
//--------------------------------------------------------------------------------------------------
template <unsigned Degree, basfld::element T> struct polynomial_mapper {
  using value_type = std::array<T, Degree + 1u>;

  CUDA_CALLABLE
  value_type map_index(unsigned index) const noexcept {
    value_type res;
    this->map_index(res, index);
    return res;
  }

  CUDA_CALLABLE
  void map_index(value_type& p, unsigned index) const noexcept {
    if (index + split < n) {
      expand_products<T>(p, mles + index, n, split, {product_terms, Degree});
    } else {
      partial_expand_products<T>(p, mles + index, n, {product_terms, Degree});
    }
  }

  const T* __restrict__ mles;
  const unsigned* __restrict__ product_terms;
  unsigned split;
  unsigned n;
};
} // namespace sxt::prfsk2
