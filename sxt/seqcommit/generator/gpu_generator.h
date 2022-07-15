#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::rstt {
class compressed_element;
}

namespace sxt::sqcgn {

//--------------------------------------------------------------------------------------------------
// cpu_get_generators
//--------------------------------------------------------------------------------------------------
void gpu_get_generators(basct::span<rstt::compressed_element> generators,
                        uint64_t offset_generators) noexcept;

} // namespace sxt::sqcgn
