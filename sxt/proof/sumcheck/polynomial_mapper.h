#pragma once

#include <array>
#include <utility>

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/proof/sumcheck/polynomial_utility.h"
#include "sxt/scalar25/operation/add.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// polynomial_mapper
//--------------------------------------------------------------------------------------------------
template <unsigned MaxDegree> struct polynomial_mapper {
  using value_type = std::array<s25t::element, MaxDegree + 1u>;

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
      (void)mult;
    }
/* void expand_products(basct::span<s25t::element> p, const s25t::element* mles, unsigned n, */
/*                      unsigned step, basct::cspan<unsigned> terms) noexcept; */
    (void)p;
    (void)index;
  }

  const s25t::element* __restrict__ mles;
  const std::pair<s25t::element, unsigned>* __restrict__ product_table;
  const unsigned* __restrict__ product_terms;
  unsigned num_products;
  unsigned mid;
  unsigned n;
};
} // namespace sxt::prfsk
