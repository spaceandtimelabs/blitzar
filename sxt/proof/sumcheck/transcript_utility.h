#pragma once

#include "sxt/base/container/span.h"

namespace sxt::prft { class transcript; }
namespace sxt::s25t { class element; }

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// init_transcript 
//--------------------------------------------------------------------------------------------------
void init_transcript(prft::transcript& transcript, unsigned num_variables,
                     unsigned round_degree) noexcept;

//--------------------------------------------------------------------------------------------------
// round_challenge 
//--------------------------------------------------------------------------------------------------
void round_challenge(s25t::element& r, prft::transcript& transcript,
    basct::cspan<s25t::element> polynomial) noexcept;
} // namespace sxt::prfsk
