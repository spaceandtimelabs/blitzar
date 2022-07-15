#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::mtxi {
struct clump2_descriptor;

//--------------------------------------------------------------------------------------------------
// compute_clump2_marker
//--------------------------------------------------------------------------------------------------
uint64_t compute_clump2_marker(const clump2_descriptor& descriptor, uint64_t clump_index,
                               uint64_t index1, uint64_t index2) noexcept;

uint64_t compute_clump2_marker(const clump2_descriptor& descriptor, uint64_t clump_index,
                               uint64_t index) noexcept;

//--------------------------------------------------------------------------------------------------
// unpack_clump2_marker
//--------------------------------------------------------------------------------------------------
void unpack_clump2_marker(uint64_t& clump_index, uint64_t& index1, uint64_t& index2,
                          const clump2_descriptor& descriptor, uint64_t marker) noexcept;

//--------------------------------------------------------------------------------------------------
// consume_clump2_marker
//--------------------------------------------------------------------------------------------------
uint64_t consume_clump2_marker(basct::span<uint64_t>& indexes,
                               const clump2_descriptor& descriptor) noexcept;
} // namespace sxt::mtxi
