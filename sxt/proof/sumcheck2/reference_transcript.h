#pragma once

#include "sxt/proof/sumcheck2/sumcheck_transcript.h"
#include "sxt/proof/transcript/transcript.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// reference_transcript
//--------------------------------------------------------------------------------------------------
template <basfld::element T>
class reference_transcript final : public sumcheck_transcript<T> {
public:
  explicit reference_transcript(prft::transcript& transcript) noexcept;

  void init(size_t num_variables, size_t round_degree) noexcept {
    (void)num_variables;
    (void)round_degree;
  }

  void round_challenge(T& r, basct::cspan<T> polynomial) noexcept {
    (void)r;
    (void)polynomial;
  }

private:
  prft::transcript& transcript_;
};
} // namespace sxt::prfsk
