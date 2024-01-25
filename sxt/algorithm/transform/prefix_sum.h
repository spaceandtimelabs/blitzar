#pragma once

#include "sxt/base/container/span.h"

namespace sxt::basdv { class stream; }

namespace sxt::algtr {
//--------------------------------------------------------------------------------------------------
// exclusive_prefix_sum 
//--------------------------------------------------------------------------------------------------
void exclusive_prefix_sum(basct::span<unsigned> out, basct::cspan<unsigned> in,
                          const basdv::stream& stream) noexcept;

//--------------------------------------------------------------------------------------------------
// inclusive_prefix_sum 
//--------------------------------------------------------------------------------------------------
void inclusive_prefix_sum(basct::span<unsigned> out, basct::cspan<unsigned> in,
                          const basdv::stream& stream) noexcept;
} // namespace sxt::algtr
