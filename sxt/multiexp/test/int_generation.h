#pragma once

#include <cstdint>
#include <random>

#include "sxt/base/container/span.h"

namespace sxt::mtxtst {
//--------------------------------------------------------------------------------------------------
// generate_uint64s
//--------------------------------------------------------------------------------------------------
void generate_uint64s(basct::span<uint64_t> generators, std::mt19937& rng) noexcept;
} // namespace sxt::mtxtst
