#pragma once

#include <cstdint>
#include <random>

#include "sxt/base/container/span.h"

namespace sxt::rstt {
struct compressed_element;
}

namespace sxt::mtxtst {
//--------------------------------------------------------------------------------------------------
// generate_ristretto_elements
//--------------------------------------------------------------------------------------------------
void generate_ristretto_elements(basct::span<rstt::compressed_element> generators,
                                 std::mt19937& rng) noexcept;
} // namespace sxt::mtxtst
