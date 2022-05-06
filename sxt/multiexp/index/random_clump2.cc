#include "sxt/multiexp/index/random_clump2.h"

namespace sxt::mtxi {
//--------------------------------------------------------------------------------------------------
// generate_random_clump2
//--------------------------------------------------------------------------------------------------
void generate_random_clump2(random_clump2& clump, std::mt19937& rng) noexcept {
  constexpr uint64_t max_clump_size = 10'000;
  constexpr uint64_t max_clump_index = 1'000;

  std::uniform_int_distribution<uint64_t> clump_size_dist{0, max_clump_size};
  std::uniform_int_distribution<uint64_t> clump_index_dist{0, max_clump_index};

  auto clump_size = clump_size_dist(rng);
  clump.clump_size = clump_size;
  clump.clump_index = clump_index_dist(rng);

  std::uniform_int_distribution<uint64_t> index1_dist{0, clump_size-1};
  auto index1 = index1_dist(rng);
  clump.index1 = index1;

  std::uniform_int_distribution<uint64_t> index2_dist{index1, clump_size-1};
  clump.index2 = index2_dist(rng);
}
}  // namespace sxt::mtxi
