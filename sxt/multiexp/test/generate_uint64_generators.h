#pragma once

#include <random>
#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::mtxtst {
//--------------------------------------------------------------------------------------------------
// generate_uint64_generators
//--------------------------------------------------------------------------------------------------
void generate_uint64_generators(basct::span<uint64_t> generators, std::mt19937& rng) noexcept {
    std::uniform_int_distribution<uint64_t> generators_gen;
    
    // populate the generators array
    for (size_t i = 0; i < generators.size(); ++i) {
      generators[i] = generators_gen(rng);
    }
}
}  // namespace sxt::mtxtst
