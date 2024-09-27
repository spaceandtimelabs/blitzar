#include "sxt/proof/sumcheck/transcript_utility.h"

#include "sxt/proof/transcript/transcript_utility.h"

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
} // namespace sxt::prfsk
