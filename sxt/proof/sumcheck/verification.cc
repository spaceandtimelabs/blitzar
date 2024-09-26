#include "sxt/proof/sumcheck/verification.h"

#include "sxt/scalar25/type/element.h"
#include "verification.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// verify_sumcheck_no_evaluation 
//--------------------------------------------------------------------------------------------------
bool verify_sumcheck_no_evaluation(s25t::element& expected_sum,
                                   basct::span<s25t::element> evaluation_point,
                                   prft::transcript& transcript, 
                                   basct::span<s25t::element> round_polynomials,
                                   unsigned round_degree) noexcept {
  (void)expected_sum;
  (void)evaluation_point;
  (void)transcript;
  (void)round_polynomials;
  (void)round_degree;
  return true;
}
} // namespace sxt::prfsk
