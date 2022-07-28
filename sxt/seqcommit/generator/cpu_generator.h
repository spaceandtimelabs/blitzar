#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::c21t {
struct element_p3;
}

namespace sxt::sqcgn {
void cpu_get_generators(basct::span<c21t::element_p3> generators, uint64_t offset) noexcept;

} // namespace sxt::sqcgn
