#include "sxt/proof/sumcheck/verification.h"

#include "sxt/base/error/assert.h"
#include "sxt/base/log/log.h"
#include "sxt/proof/sumcheck/polynomial_utility.h"
#include "sxt/proof/sumcheck/transcript_utility.h"
#include "sxt/proof/transcript/transcript_utility.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// verify_sumcheck_no_evaluation 
//--------------------------------------------------------------------------------------------------
bool verify_sumcheck_no_evaluation(s25t::element& expected_sum,
                                   basct::span<s25t::element> evaluation_point,
                                   prft::transcript& transcript, 
                                   basct::cspan<s25t::element> round_polynomials,
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

  init_transcript(transcript, num_variables, round_degree);

  // go through sumcheck rounds
  for (unsigned round_index=0; round_index<num_variables; ++round_index) {
    auto polynomial =
        round_polynomials.subspan((round_degree + 1u) * round_index, round_degree + 1u);

    // check sum
    s25t::element sum;
    sum_polynomial_01(sum, polynomial);
    if (expected_sum != sum) {
      basl::info("sumcheck verification failed on round {}", round_index + 1);
      return false;
    }

    // draw a random scalar
    s25t::element r;
    round_challenge(r, transcript, polynomial);
    evaluation_point[round_index] = r;


    // evaluate at random point
    evaluate_polynomial(expected_sum, polynomial, r);
  }

  return true;
}
} // namespace sxt::prfsk
