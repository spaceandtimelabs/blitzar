#pragma once

#include "sxt/proof/sumcheck2/sumcheck_transcript.h"
#include "sxt/proof/transcript/transcript.h"
#include "sxt/proof/transcript/transcript_utility.h"

namespace sxt::prfsk2 {
//--------------------------------------------------------------------------------------------------
// reference_transcript
//--------------------------------------------------------------------------------------------------
template <basfld::element T>
class reference_transcript final : public sumcheck_transcript<T> {
public:
  explicit reference_transcript(prft::transcript& transcript) noexcept;

  void init(size_t num_variables, size_t round_degree) noexcept {
    prft::set_domain(transcript_, "sumcheck proof v1");
    prft::append_value(transcript_, "n", num_variables);
    prft::append_value(transcript_, "k", round_degree);
  }

  void round_challenge(T& r, basct::cspan<T> polynomial) noexcept {
    prft::append_values(transcript_, "P", polynomial);
    prft::challenge_value(r, transcript_, "R");
  }

private:
  prft::transcript& transcript_;
};
} // namespace sxt::prfsk2
