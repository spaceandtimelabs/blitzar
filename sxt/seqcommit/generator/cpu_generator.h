#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::rstt {
class compressed_element;
}

namespace sxt::c21t {
struct element_p3;
}

namespace sxt::sqcgn {

//--------------------------------------------------------------------------------------------------
// cpu_get_generators
//--------------------------------------------------------------------------------------------------
void cpu_get_generators(basct::span<rstt::compressed_element> generators,
                        uint64_t offset_generators) noexcept;

void cpu_get_generators(basct::span<c21t::element_p3> generators, uint64_t offset) noexcept;

} // namespace sxt::sqcgn
