#include "sxt/proof/sumcheck/reference_transcript.h"

#include "sxt/proof/transcript/transcript_utility.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// init
//--------------------------------------------------------------------------------------------------
void reference_transcript::init(size_t num_variables, size_t round_degree) noexcept {
  (void)num_variables;
  (void)round_degree;
  (void)transcript_;
  /* prft::set_domain(transcript, "sumcheck proof v1"); */
  /* prft::append_value(transcript, "n", num_variables); */
  /* prft::append_value(transcript, "k", round_degree); */
}

//--------------------------------------------------------------------------------------------------
// round_challenge
//--------------------------------------------------------------------------------------------------
void reference_transcript::round_challenge(s25t::element& r,
                                           basct::cspan<s25t::element> polynomial) noexcept {
  (void)r;
  (void)polynomial;
#if 0
  prft::append_values(transcript, "P", polynomial);
  prft::challenge_value(r, transcript, "R");
#endif
}
} // namespace sxt::prfsk
