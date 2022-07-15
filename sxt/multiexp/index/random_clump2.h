#pragma once

#include <cstdint>
#include <random>

#include "sxt/multiexp/index/clump2_descriptor.h"

namespace sxt::mtxi {
//--------------------------------------------------------------------------------------------------
// random_clump2
//--------------------------------------------------------------------------------------------------
struct random_clump2 {
  uint64_t clump_size;
  uint64_t clump_index;
  uint64_t index1;
  uint64_t index2;
};

//--------------------------------------------------------------------------------------------------
// generate_random_clump2
//--------------------------------------------------------------------------------------------------
void generate_random_clump2(random_clump2& clump, std::mt19937& rng) noexcept;
} // namespace sxt::mtxi
