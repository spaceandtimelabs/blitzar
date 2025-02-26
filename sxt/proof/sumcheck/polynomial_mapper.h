/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2025-present Space and Time Labs, Inc.
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

#include "sxt/base/field/element.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/proof/sumcheck/polynomial_utility.h"

namespace sxt::prfsk {
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
} // namespace sxt::prfsk
