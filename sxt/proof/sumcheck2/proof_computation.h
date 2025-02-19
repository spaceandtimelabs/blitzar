#pragma once

#include "sxt/base/error/assert.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/base/field/element.h"
#include "sxt/proof/sumcheck2/driver.h"
#include "sxt/proof/sumcheck2/sumcheck_transcript.h"
#include "sxt/execution/async/future.h"

namespace sxt::prfsk2 {
//--------------------------------------------------------------------------------------------------
// prove_sum
//--------------------------------------------------------------------------------------------------
template <basfld::element T>
xena::future<> prove_sum(basct::span<T> polynomials,
                         basct::span<T> evaluation_point,
                         sumcheck_transcript<T>& transcript, const driver<T>& drv,
                         basct::cspan<T> mles,
                         basct::cspan<std::pair<T, unsigned>> product_table,
                         basct::cspan<unsigned> product_terms, unsigned n) noexcept {
  SXT_RELEASE_ASSERT(0 < n);
  auto num_variables = std::max(basn::ceil_log2(n), 1);
  auto polynomial_length = polynomials.size() / num_variables;
  auto num_mles = mles.size() / n;
  SXT_RELEASE_ASSERT(
      // clang-format off
      polynomial_length > 1 &&
      evaluation_point.size() == num_variables &&
      polynomials.size() == num_variables * polynomial_length &&
      mles.size() == n * num_mles
      // clang-format on
  );

  transcript.init(num_variables, polynomial_length - 1);

  auto ws = co_await drv.make_workspace(mles, product_table, product_terms, n);

  for (unsigned round_index = 0; round_index < num_variables; ++round_index) {
    auto polynomial = polynomials.subspan(round_index * polynomial_length, polynomial_length);

    // compute the round polynomial
    co_await drv.sum(polynomial, *ws);

    // draw the next random challenge
    T r;
    transcript.round_challenge(r, polynomial);
    evaluation_point[round_index] = r;

    // fold the polynomial
    if (round_index < num_variables - 1u) {
      co_await drv.fold(*ws, r);
    }
  }
}
} // namespace sxt:prfsk2
