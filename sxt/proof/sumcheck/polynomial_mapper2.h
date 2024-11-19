#pragma once

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/proof/sumcheck/polynomial_utility.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// polynomial_mapper
//--------------------------------------------------------------------------------------------------
template <unsigned Degree> struct polynomial_mapper2 {
  using value_type = std::array<s25t::element, Degree + 1u>;

  CUDA_CALLABLE
  value_type map_index(unsigned index) const noexcept {
    value_type res;
    this->map_index(res, index);
    return res;
  }

  CUDA_CALLABLE
  void map_index(value_type& p, unsigned index) const noexcept {
    return;
    if (index + split < n) {
      expand_products(p, mles, n, split, {product_terms, Degree});
    } else {
      partial_expand_products(p, mles, n, {product_terms, Degree});
    }
  }

  const s25t::element* __restrict__ mles;
  const unsigned* __restrict__ product_terms;
  unsigned split;
  unsigned n;
};
} // namespace sxt::prfsk
