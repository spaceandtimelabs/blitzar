#include "sxt/proof/sumcheck/proof_computation.h"

#include "sxt/base/error/assert.h"
#include "sxt/proof/sumcheck/transcript_utility.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// prove_sum 
//--------------------------------------------------------------------------------------------------
xena::future<> prove_sum(basct::span<s25t::element> polynomials, prft::transcript& transcript,
                         basct::cspan<s25t::element> mles,
                         basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                         basct::cspan<unsigned> product_terms, unsigned n) noexcept {
/* void init_transcript(prft::transcript& transcript, unsigned num_variables, */
/*                      unsigned round_degree) noexcept; */
/*  */
  (void)n;
  (void)polynomials;
  (void)transcript;
  (void)mles;
  (void)product_terms;
  (void)product_table;
  return {};
}
} // namespace sxt::prfsk
