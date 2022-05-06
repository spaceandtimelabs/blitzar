#pragma once

#include <cstdint>

namespace sxt::mtxi {
struct clump2_descriptor;

//--------------------------------------------------------------------------------------------------
// init_clump2_descriptor
//--------------------------------------------------------------------------------------------------
void init_clump2_descriptor(clump2_descriptor& descriptor,
                            uint64_t clump_size) noexcept;
}  // namespace sxt::mtxi
