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
void apply_marker_row_transformation(basct::span<uint64_t>& indexes, F consumer,
                                     size_t offset) noexcept {
  auto out = indexes.begin() + offset;
  auto rest = indexes.subspan(offset);
  while (!rest.empty()) {
    *out++ = consumer(rest);
  }
  indexes = basct::span<uint64_t>{indexes.data(),
                                  static_cast<size_t>(std::distance(indexes.begin(), out))};
}

//--------------------------------------------------------------------------------------------------
// apply_marker_transformation
//--------------------------------------------------------------------------------------------------
template <class F, class OffsetFunctor>
size_t apply_marker_transformation(basct::span<basct::span<uint64_t>> rows, F consumer,
                                   OffsetFunctor offset_functor) noexcept {
  size_t count = 0;
  for (auto& row : rows) {
    auto offset = offset_functor(row);
    apply_marker_row_transformation(row, consumer, offset);
    count += row.size() - offset;
  }
  return count;
}

template <class F>
size_t apply_marker_transformation(basct::span<basct::span<uint64_t>> rows, F consumer) noexcept {
  return apply_marker_transformation(rows, consumer,
                                     [](basct::cspan<uint64_t> /*row*/) noexcept { return 0; });
}

} // namespace sxt::mtxi
