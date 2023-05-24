/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
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
#include "sxt/proof/inner_product/verification_computation.h"

#include <vector>

#include "sxt/base/error/assert.h"
#include "sxt/scalar25/operation/inner_product.h"
#include "sxt/scalar25/operation/inv.h"
#include "sxt/scalar25/operation/mul.h"
#include "sxt/scalar25/operation/neg.h"
#include "sxt/scalar25/operation/sq.h"

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// compute_g_exponents
//--------------------------------------------------------------------------------------------------
void compute_g_exponents(basct::span<s25t::element> g_exponents, const s25t::element& allinv,
                         const s25t::element& ap_value,
                         basct::cspan<s25t::element> x_sq_vector) noexcept {
  auto n = g_exponents.size();
  s25o::mul(g_exponents[0], allinv, ap_value);
  size_t a = 1;
  size_t b = 2;
  auto multiplier_iter = x_sq_vector.data() + x_sq_vector.size();
  while (a != n) {
    auto& multiplier = *--multiplier_iter;
    for (size_t i = a; i < b; ++i) {
      s25o::mul(g_exponents[i], multiplier, g_exponents[i - a]);
    }
    a = b;
    b = 2 * a;
  }
}

//--------------------------------------------------------------------------------------------------
// compute_lr_exponents_part1
//--------------------------------------------------------------------------------------------------
void compute_lr_exponents_part1(basct::span<s25t::element> l_exponents,
                                basct::span<s25t::element> r_exponents, s25t::element& allinv,
                                basct::cspan<s25t::element> x_vector) noexcept {
  auto num_rounds = l_exponents.size();

  // 0
  s25o::inv(allinv, x_vector[0]);
  s25o::sq(l_exponents[0], x_vector[0]);
  s25o::sq(r_exponents[0], allinv);
  s25o::neg(r_exponents[0], r_exponents[0]);

  // 1 .. num_rounds-1
  for (size_t i = 1; i < num_rounds; ++i) {
    auto& xi = x_vector[i];
    s25t::element xi_inv;
    s25o::inv(xi_inv, xi);
    s25o::mul(allinv, allinv, xi_inv);

    // li
    s25o::sq(l_exponents[i], xi);

    // ri
    s25o::sq(r_exponents[i], xi_inv);
    s25o::neg(r_exponents[i], r_exponents[i]);
  }
}

//--------------------------------------------------------------------------------------------------
// compute_verification_exponents
//--------------------------------------------------------------------------------------------------
void compute_verification_exponents(basct::span<s25t::element> exponents,
                                    basct::cspan<s25t::element> x_vector,
                                    const s25t::element& ap_value,
                                    basct::cspan<s25t::element> b_vector) noexcept {
  auto num_exponents = exponents.size();
  auto num_rounds = x_vector.size();
  auto n = b_vector.size();
  auto np = 1ull << num_rounds;
  // clang-format off
  SXT_DEBUG_ASSERT(
      n > 0 &&
      (n == np || n > (1ull << (num_rounds-1))) &&
      num_exponents == 1 + np + 2 * num_rounds &&
      exponents.size() == num_exponents &&
      x_vector.size() == num_rounds &&
      b_vector.size() == n
  );
  // clang-format on

  auto& product = exponents[0];
  auto g_exponents = exponents.subspan(1, np);
  auto l_exponents = exponents.subspan(1 + np, num_rounds);
  auto r_exponents = exponents.subspan(1 + np + num_rounds, num_rounds);

  if (n == 1) {
    s25o::mul(product, b_vector[0], ap_value);
    g_exponents[0] = ap_value;
    return;
  }

  s25t::element allinv;
  compute_lr_exponents_part1(l_exponents, r_exponents, allinv, x_vector);
  compute_g_exponents(g_exponents, allinv, ap_value, l_exponents);
  s25o::inner_product(product, g_exponents, b_vector);

  for (auto& li : l_exponents) {
    s25o::neg(li, li);
  }
}
} // namespace sxt::prfip
