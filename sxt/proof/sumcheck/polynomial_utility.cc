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
#include "sxt/proof/sumcheck/polynomial_utility.h"

#include <cassert>

#include "sxt/scalar25/operation/add.h"
#include "sxt/scalar25/operation/mul.h"
#include "sxt/scalar25/operation/muladd.h"
#include "sxt/scalar25/operation/sub.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// sum_polynomial_01
//--------------------------------------------------------------------------------------------------
void sum_polynomial_01(s25t::element& e, basct::cspan<s25t::element> polynomial) noexcept {
  if (polynomial.empty()) {
    e = s25t::element{};
  }
  e = polynomial[0];
  for (unsigned i = 1; i < polynomial.size(); ++i) {
    s25o::add(e, e, polynomial[i]);
  }
}

//--------------------------------------------------------------------------------------------------
// evaluate_polynomial
//--------------------------------------------------------------------------------------------------
void evaluate_polynomial(s25t::element& e, basct::cspan<s25t::element> polynomial,
                         const s25t::element& x) noexcept {
  if (polynomial.empty()) {
    e = s25t::element{};
  }
  auto i = polynomial.size();
  --i;
  e = polynomial[i];
  while (i > 0) {
    --i;
    s25o::muladd(e, e, x, polynomial[i]);
  }
}

//--------------------------------------------------------------------------------------------------
// expand_products
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void expand_products(basct::span<s25t::element> p, const s25t::element* mles, unsigned n,
                     unsigned step, basct::cspan<unsigned> terms) noexcept {
  auto num_terms = terms.size();
  assert(num_terms > 0 && p.size() == num_terms + 1u);
  s25t::element a, b;
  auto mle_index = terms[0];
  a = *(mles + mle_index * n);
  b = *(mles + mle_index * n + step);
  s25o::sub(b, b, a);
  p[0] = a;
  p[1] = b;

  for (unsigned i = 1; i < num_terms; ++i) {
    auto mle_index = terms[i];
    a = *(mles + mle_index * n);
    b = *(mles + mle_index * n + step);
    s25o::sub(b, b, a);

    auto c_prev = p[0];
    s25o::mul(p[0], c_prev, a);
    for (unsigned pow = 1u; pow < i + 1u; ++pow) {
      auto c = p[pow];
      s25o::mul(p[pow], c, a);
      s25o::muladd(p[pow], c_prev, b, p[pow]);
      c_prev = c;
    }
    s25o::mul(p[i + 1u], c_prev, b);
  }
}
} // namespace sxt::prfsk
