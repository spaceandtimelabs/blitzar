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
#include "sxt/proof/sumcheck/sumcheck_random.h"

#include <random>

#include "sxt/base/error/assert.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/scalar25/random/element.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk2 {
//--------------------------------------------------------------------------------------------------
// generate_random_sumcheck_problem
//--------------------------------------------------------------------------------------------------
void generate_random_sumcheck_problem(
    std::vector<s25t::element>& mles,
    std::vector<std::pair<s25t::element, unsigned>>& product_table,
    std::vector<unsigned>& product_terms, unsigned& n, basn::fast_random_number_generator& rng,
    const random_sumcheck_descriptor& descriptor) noexcept {
  std::mt19937 rng_p{rng()};

  // n
  SXT_RELEASE_ASSERT(descriptor.min_length <= descriptor.max_length);
  std::uniform_int_distribution<unsigned> n_dist{descriptor.min_length, descriptor.max_length};
  n = n_dist(rng_p);

  // num_mles
  SXT_RELEASE_ASSERT(descriptor.min_num_mles <= descriptor.max_num_mles);
  std::uniform_int_distribution<unsigned> num_mles_dist{descriptor.min_num_mles,
                                                        descriptor.max_num_mles};
  auto num_mles = num_mles_dist(rng_p);

  // num_products
  SXT_RELEASE_ASSERT(descriptor.min_num_products <= descriptor.max_num_products);
  std::uniform_int_distribution<unsigned> num_products_dist{descriptor.min_num_products,
                                                            descriptor.max_num_products};
  auto num_products = num_products_dist(rng_p);

  // mles
  mles.resize(n * num_mles);
  s25rn::generate_random_elements(mles, rng);

  // product_table
  unsigned num_terms = 0;
  product_table.resize(num_products);
  SXT_RELEASE_ASSERT(descriptor.min_product_length <= descriptor.max_product_length);
  std::uniform_int_distribution<unsigned> product_length_dist{descriptor.min_product_length,
                                                              descriptor.max_product_length};
  for (auto& [s, len] : product_table) {
    s25rn::generate_random_element(s, rng);
    len = product_length_dist(rng_p);
    num_terms += len;
  }

  // product_terms
  product_terms.resize(num_terms);
  std::uniform_int_distribution<unsigned> mle_dist{0, num_mles - 1};
  for (auto& term : product_terms) {
    term = mle_dist(rng_p);
  }
}
} // namespace sxt::prfsk2
