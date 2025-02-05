#include "sxt/proof/sumcheck/reference_transcript.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// init
//--------------------------------------------------------------------------------------------------
void reference_transcript::init(size_t num_variables, size_t round_degree) noexcept {
  (void)num_variables;
  (void)round_degree;
}

//--------------------------------------------------------------------------------------------------
// round_challenge
//--------------------------------------------------------------------------------------------------
void reference_transcript::round_challenge(s25t::element& r,
                                           basct::cspan<s25t::element> polynomial) noexcept {
  (void)r;
  (void)polynomial;
}
} // namespace sxt::prfsk
