#include "sxt/proof/sumcheck/reference_transcript.h"

#include "sxt/proof/transcript/transcript_utility.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
reference_transcript::reference_transcript(prft::transcript& transcript) noexcept
    : transcript_{transcript} {}

//--------------------------------------------------------------------------------------------------
// init
//--------------------------------------------------------------------------------------------------
void reference_transcript::init(size_t num_variables, size_t round_degree) noexcept {
  prft::set_domain(transcript_, "sumcheck proof v1");
  prft::append_value(transcript_, "n", num_variables);
  prft::append_value(transcript_, "k", round_degree);
}

//--------------------------------------------------------------------------------------------------
// round_challenge
//--------------------------------------------------------------------------------------------------
void reference_transcript::round_challenge(s25t::element& r,
                                           basct::cspan<s25t::element> polynomial) noexcept {
  prft::append_values(transcript_, "P", polynomial);
  prft::challenge_value(r, transcript_, "R");
}
} // namespace sxt::prfsk
