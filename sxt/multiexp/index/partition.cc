#include "sxt/multiexp/index/partition.h"

#include <iterator>

#include "sxt/multiexp/index/partition_marker_utility.h"

namespace sxt::mtxi {
//--------------------------------------------------------------------------------------------------
// partition_row
//--------------------------------------------------------------------------------------------------
void partition_row(basct::span<uint64_t>& indexes, uint64_t partition_size) noexcept {
  auto out = indexes.begin();
  auto rest = indexes;
  while(!rest.empty()) {
    *out++ = consume_partition_marker(rest, partition_size);
  }
  indexes = basct::span<uint64_t>{
      indexes.data(), static_cast<size_t>(std::distance(indexes.begin(), out))};
}

//--------------------------------------------------------------------------------------------------
// partition_rows
//--------------------------------------------------------------------------------------------------
size_t partition_rows(basct::span<basct::span<uint64_t>> rows,
                     uint64_t partition_size) noexcept {
  size_t count = 0;
  for (auto& row : rows) {
    partition_row(row, partition_size);
    count += row.size();
  }
  return count;
}
}  // namespace sxt::mtxi
