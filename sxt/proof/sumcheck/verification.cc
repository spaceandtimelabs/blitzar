#include "sxt/proof/sumcheck/verification.h"

#include "sxt/base/log/log.h"
#include "sxt/proof/sumcheck/polynomial_utility.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// verify_sumcheck_no_evaluation 
//--------------------------------------------------------------------------------------------------
bool verify_sumcheck_no_evaluation(s25t::element& expected_sum,
                                   basct::span<s25t::element> evaluation_point,
                                   prft::transcript& transcript, 
                                   basct::span<s25t::element> round_polynomials,
                                   unsigned round_degree) noexcept {
  auto num_variables = evaluation_point.size();
  basl::info("verifying sumcheck of {} variables and round degree {}", num_variables, round_degree);

  // check dimensions
  if (auto expected_count = (round_degree + 1u) * num_variables;
      round_polynomials.size() != expected_count) {
    basl::info("sumcheck verification failed: expected {} scalars for round_polynomials but got {}",
               expected_count, round_polynomials.size());
    return false;
  }

  // go through sumcheck rounds
  for (unsigned round_index=0; round_index<num_variables; ++round_index) {
    auto round_polynomial =
        round_polynomials.subspan((round_degree + 1u) * round_index, round_degree + 1u);

    // TODO: commit to round polynomial
    // TODO: draw evaluation point

    // check sum
    s25t::element round_sum;
    sum_polynomial_01(round_sum, round_polynomial);
    if (expected_sum != round_sum) {
      basl::info("sumcheck verification failed on round {}", round_index + 1);
      return false;
    }

    // evaluate at random point
    /* evaluate_polynomial(expected_sum, round_polynomial, evaluation_point[round_index]); */
  }

  return true;
}
} // namespace sxt::prfsk
