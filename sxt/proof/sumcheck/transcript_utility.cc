#include "sxt/proof/sumcheck/transcript_utility.h"

#include "sxt/proof/transcript/transcript_utility.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// init_transcript 
//--------------------------------------------------------------------------------------------------
void init_transcript(prft::transcript& transcript, unsigned num_variables,
                     unsigned round_degree) noexcept {
  prft::set_domain(transcript, "sumcheck proof v1");
  prft::append_value(transcript, "n", num_variables);
  prft::append_value(transcript, "k", round_degree);
}

//--------------------------------------------------------------------------------------------------
// round_challenge 
//--------------------------------------------------------------------------------------------------
void round_challenge(s25t::element& r, prft::transcript& transcript,
                     basct::cspan<s25t::element> polynomial) noexcept {
  prft::append_values(transcript, "P", polynomial);
  prft::challenge_value(r, transcript, "R");
}
} // namespace sxt::prfsk
