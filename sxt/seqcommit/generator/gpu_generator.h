#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::sqcb { class commitment; }

namespace sxt::sqcgn {

//--------------------------------------------------------------------------------------------------
// cpu_get_generators
//--------------------------------------------------------------------------------------------------
void gpu_get_generators(
    basct::span<sqcb::commitment> generators,
    uint64_t offset_generators
) noexcept;

} // namespace sxt::sqcgn
