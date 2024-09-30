#include "sxt/proof/sumcheck/proof_computation.h"

#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// prove_sum 
//--------------------------------------------------------------------------------------------------
xena::future<> prove_sum(basct::span<s25t::element> polynomials, prft::transcript& transcript,
                         basct::cspan<s25t::element> mles, basct::cspan<unsigned> product_terms,
                         basct::cspan<std::pair<s25t::element, unsigned>> product_table) noexcept {
  (void)polynomials;
  (void)transcript;
  (void)mles;
  (void)product_terms;
  (void)product_table;
  return {};
}
} // namespace sxt::prfsk
