#pragma once

#include <array>
#include <utility>

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/proof/sumcheck/polynomial_utility.h"
#include "sxt/scalar25/operation/add.h"
#include "sxt/scalar25/operation/muladd.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// polynomial_mapper
//--------------------------------------------------------------------------------------------------
template <unsigned MaxDegree> struct polynomial_mapper {
  using value_type = std::array<s25t::element, MaxDegree + 1u>;

  CUDA_CALLABLE
  value_type map_index(unsigned index) const noexcept {
    value_type res;
    this->map_index(res, index);
    return res;
  }

  CUDA_CALLABLE
  void map_index(value_type& p, unsigned index) const noexcept {
    // zero
    for (auto& pi : p) { 
      pi = {};
    }

    // expand products
    auto mle_data = mles + index;
    auto terms_data = product_terms;
    s25t::element prod[MaxDegree + 1u];
    for (unsigned product_index = 0; product_index < num_products; ++product_index) {
      auto [mult, num_terms] = product_table[product_index];

      expand_products({prod, num_terms + 1u}, mle_data, n, mid, {terms_data, num_terms});
      terms_data += num_terms;

      for (unsigned i=0; i<num_terms+1; ++i) {
        s25o::muladd(p[i], mult, prod[i], p[i]);
      }
    }
  }

  const s25t::element* __restrict__ mles;
  const std::pair<s25t::element, unsigned>* __restrict__ product_table;
  const unsigned* __restrict__ product_terms;
  unsigned num_products;
  unsigned mid;
  unsigned n;
};
} // namespace sxt::prfsk
