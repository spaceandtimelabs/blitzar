#include "sxt/multiexp/bitset_multiprod/value_cache_utility.h"

namespace sxt::mtxbmp {
//--------------------------------------------------------------------------------------------------
// compute_cache_size
//--------------------------------------------------------------------------------------------------
size_t compute_cache_size(size_t num_terms) noexcept {
  auto left_size = num_terms / 2;
  auto right_size = num_terms - left_size;
  return (1 << left_size) + (1 << right_size) - 2;
}
} // namespace sxt::mtxbmp
