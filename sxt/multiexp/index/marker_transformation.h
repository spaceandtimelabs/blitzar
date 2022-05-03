#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>

#include "sxt/base/container/span.h"

namespace sxt::mtxi {
//--------------------------------------------------------------------------------------------------
// apply_marker_row_transformation
//--------------------------------------------------------------------------------------------------
template <class F>
void apply_marker_row_transformation(basct::span<uint64_t>& indexes,
                                     F consumer) noexcept {
  auto out = indexes.begin();
  auto rest = indexes;
  while(!rest.empty()) {
    *out++ = consumer(rest);
  }
  indexes = basct::span<uint64_t>{
      indexes.data(), static_cast<size_t>(std::distance(indexes.begin(), out))};
}

//--------------------------------------------------------------------------------------------------
// apply_marker_transformation
//--------------------------------------------------------------------------------------------------
template <class F>
size_t apply_marker_transformation(basct::span<basct::span<uint64_t>> rows,
                                   F consumer) noexcept {
  size_t count = 0;
  for (auto& row : rows) {
    apply_marker_row_transformation(row, consumer);
    count += row.size();
  }
  return count;
}
}  // namespace sxt::mtxi
