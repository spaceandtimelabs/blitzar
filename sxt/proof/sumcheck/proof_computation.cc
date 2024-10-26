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
#include "sxt/proof/sumcheck/proof_computation.h"

#include "sxt/base/error/assert.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/proof/sumcheck/driver.h"
#include "sxt/proof/sumcheck/transcript_utility.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// prove_sum
//--------------------------------------------------------------------------------------------------
xena::future<> prove_sum(basct::span<s25t::element> polynomials,
                         basct::span<s25t::element> evaluation_point, prft::transcript& transcript,
                         const driver& drv, basct::cspan<s25t::element> mles,
                         basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                         basct::cspan<unsigned> product_terms, unsigned n) noexcept {
  SXT_RELEASE_ASSERT(0 < n);
  auto num_variables = std::max(basn::ceil_log2(n), 1);
  auto polynomial_length = polynomials.size() / num_variables;
  auto num_mles = mles.size() / n;
  SXT_RELEASE_ASSERT(
      // clang-format off
      polynomial_length > 1 &&
      polynomials.size() == num_variables * polynomial_length &&
      mles.size() == n * num_mles
      // clang-format on
  );

  init_transcript(transcript, num_variables, polynomial_length - 1);

  auto ws = co_await drv.make_workspace(mles, product_table, product_terms, n);

  for (unsigned round_index = 0; round_index < num_variables; ++round_index) {
    auto polynomial = polynomials.subspan(round_index * polynomial_length, polynomial_length);

    // compute the round polynomial
    co_await drv.sum(polynomial, *ws);

    // draw the next random challenge
    s25t::element r;
    round_challenge(r, transcript, polynomial);
    evaluation_point[round_index] = r;

    // fold the polynomial
    co_await drv.fold(*ws, r);
  }
}
} // namespace sxt::prfsk
