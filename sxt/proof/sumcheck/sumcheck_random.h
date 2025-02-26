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

#include <utility>
#include <vector>

#include "sxt/proof/sumcheck/constant.h"

namespace sxt::s25t {
class element;
}
namespace sxt::basn {
class fast_random_number_generator;
}

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// random_sumcheck_descriptor
//--------------------------------------------------------------------------------------------------
struct random_sumcheck_descriptor {
  unsigned min_length = 1;
  unsigned max_length = 10;

  unsigned min_num_products = 1;
  unsigned max_num_products = 5;

  unsigned min_product_length = 2;
  unsigned max_product_length = max_degree_v;

  unsigned min_num_mles = 1;
  unsigned max_num_mles = 5;
};

//--------------------------------------------------------------------------------------------------
// generate_random_sumcheck_problem
//--------------------------------------------------------------------------------------------------
void generate_random_sumcheck_problem(
    std::vector<s25t::element>& mles,
    std::vector<std::pair<s25t::element, unsigned>>& product_table,
    std::vector<unsigned>& product_terms, unsigned& n, basn::fast_random_number_generator& rng,
    const random_sumcheck_descriptor& descriptor) noexcept;
} // namespace sxt::prfsk
