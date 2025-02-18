#pragma once

#include "sxt/base/container/span.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/log/log.h"
#include "sxt/proof/sumcheck2/sumcheck_transcript.h"
#include "sxt/proof/sumcheck2/polynomial_utility.h"

namespace sxt::prfsk2 {
//--------------------------------------------------------------------------------------------------
// verify_sumcheck_no_evaluation
//--------------------------------------------------------------------------------------------------
template <basfld::element T>
bool verify_sumcheck_no_evaluation(T& expected_sum,
                                   basct::span<T> evaluation_point,
                                   sumcheck_transcript<T>& transcript,
                                   basct::cspan<T> round_polynomials,
                                   unsigned round_degree) noexcept {
  auto num_variables = evaluation_point.size();
  SXT_RELEASE_ASSERT(
      // clang-format off
      num_variables > 0 && round_degree > 0
      // clang-format on
  );

  basl::info("verifying sumcheck of {} variables and round degree {}", num_variables, round_degree);

  // check dimensions
  if (auto expected_count = (round_degree + 1u) * num_variables;
      round_polynomials.size() != expected_count) {
    basl::info("sumcheck verification failed: expected {} scalars for round_polynomials but got {}",
               expected_count, round_polynomials.size());
    return false;
  }

  transcript.init(num_variables, round_degree);

  // go through sumcheck rounds
  for (unsigned round_index = 0; round_index < num_variables; ++round_index) {
    auto polynomial =
        round_polynomials.subspan((round_degree + 1u) * round_index, round_degree + 1u);

    // check sum
    T sum;
    sum_polynomial_01(sum, polynomial);
    if (expected_sum != sum) {
      basl::info("sumcheck verification failed on round {}", round_index + 1);
      return false;
    }

    // draw a random scalar
    T r;
    transcript.round_challenge(r, polynomial);
    evaluation_point[round_index] = r;

    // evaluate at random point
    evaluate_polynomial(expected_sum, polynomial, r);
  }

  return true;
}
} // namespace sxt::prfsk2
