#include "sxt/multiexp/index/partition_marker_utility.h"

#include <cassert>

#include "sxt/base/bit/count.h"

namespace sxt::mtxi {
//--------------------------------------------------------------------------------------------------
// consume_partition_marker
//--------------------------------------------------------------------------------------------------
uint64_t consume_partition_marker(basct::span<uint64_t>& indexes,
                                  uint64_t partition_size) noexcept {
  assert(!indexes.empty());
  assert(partition_size > 0 && partition_size < 64);
  auto n = indexes.size();
  auto idx = indexes[0];
  auto partition_index = idx / partition_size;
  auto partition_first = partition_index * partition_size;
  auto partition_offset = idx - partition_index * partition_size;
  uint64_t marker =
      (partition_index << partition_size) | (1 << partition_offset);
  size_t i = 1;
  for (; i<n; ++i) {
    assert(idx < indexes[i]);
    idx = indexes[i];
    partition_offset = idx - partition_first; 
    if (partition_offset >= partition_size) {
      break;
    }
    marker |= (1 << partition_offset);
  }
  indexes = basct::span{indexes.data() + i, n - i};
  return marker;
}
}  // namespace sxt::mtxi
