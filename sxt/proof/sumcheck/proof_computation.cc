#include "sxt/proof/sumcheck/proof_computation.h"

#include "sxt/base/error/assert.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/execution/async/coroutine.h"
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
  auto num_variables = basn::ceil_log2(n);
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

  auto ws = drv.make_workspace(mles, product_table, product_terms);

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
