#include "sxt/multiexp/random/int_generation.h"

namespace sxt::mtxrn {
//--------------------------------------------------------------------------------------------------
// generate_uint64s
//--------------------------------------------------------------------------------------------------
void generate_uint64s(basct::span<uint64_t> generators, std::mt19937& rng) noexcept {
  std::uniform_int_distribution<uint64_t> generators_gen;

  // populate the generators array
  for (size_t i = 0; i < generators.size(); ++i) {
    generators[i] = generators_gen(rng);
  }
}
} // namespace sxt::mtxrn
