#pragma once

#include "sxt/proof/sumcheck/sumcheck_transcript.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// reference_transcript
//--------------------------------------------------------------------------------------------------
class reference_transcript final : public sumcheck_transcript {
 public:
   void init(size_t num_variables, size_t round_degree) noexcept override;

   void round_challenge(s25t::element& r, basct::cspan<s25t::element> polynomial) noexcept override;

   /* virtual void init(size_t num_variables, size_t round_degree) noexcept = 0; */
   /*  */
   /* virtual void round_challenge(s25t::element& r, */
   /*                              basct::cspan<s25t::element> polynomial) noexcept = 0; */
 private:
};
} // namespace sxt::prfsk
