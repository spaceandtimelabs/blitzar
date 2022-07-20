#pragma once

#include <cstdint>
#include <random>

#include "sxt/base/container/span.h"

namespace sxt::c21t {
struct element_p3;
}

namespace sxt::mtxtst {
//--------------------------------------------------------------------------------------------------
// generate_curve21_elements
//--------------------------------------------------------------------------------------------------
void generate_curve21_elements(basct::span<c21t::element_p3> generators,
                               std::mt19937& rng) noexcept;
} // namespace sxt::mtxtst
