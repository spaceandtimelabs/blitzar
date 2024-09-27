#pragma once

namespace sxt::prft { class transcript; }

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// init_transcript 
//--------------------------------------------------------------------------------------------------
void init_transcript(prft::transcript& transcript, unsigned num_variables,
                     unsigned round_degree) noexcept;
} // namespace sxt::prfsk
