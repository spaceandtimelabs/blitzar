/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <array>
#include <cassert>
#include <utility>

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/proof/sumcheck/polynomial_utility.h"
#include "sxt/scalar25/operation/add.h"
#include "sxt/scalar25/operation/mul.h"
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
    auto mle_data = mles + index;
    auto terms_data = product_terms;
    s25t::element prod[MaxDegree + 1u];

    // first iteration
    assert(num_products > 0);
    auto [mult, num_terms] = product_table[0];
    expand_products({prod, num_terms + 1u}, mle_data, n, mid, {terms_data, num_terms});
    terms_data += num_terms;
    for (unsigned i = 0; i < num_terms + 1; ++i) {
      s25o::mul(p[i], mult, prod[i]);
    }

    // remaining iterations
    for (unsigned product_index = 1; product_index < num_products; ++product_index) {
      auto [mult, num_terms] = product_table[product_index];
      expand_products({prod, num_terms + 1u}, mle_data, n, mid, {terms_data, num_terms});
      terms_data += num_terms;
      for (unsigned i = 0; i < num_terms + 1; ++i) {
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
