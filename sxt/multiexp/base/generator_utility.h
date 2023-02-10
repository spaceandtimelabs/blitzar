#pragma once

#include <algorithm>

#include "sxt/base/container/blob_array.h"
#include "sxt/base/container/span.h"
#include "sxt/base/error/assert.h"

namespace sxt::mtxb {
//--------------------------------------------------------------------------------------------------
// filter_generators
//--------------------------------------------------------------------------------------------------
template <class T>
void filter_generators(basct::span<T> dst, basct::cspan<T> src,
                       const basct::blob_array& masks) noexcept {
  SXT_DEBUG_ASSERT(dst.size() <= src.size() && src.size() == masks.size());
  if (dst.size() == src.size()) {
    std::copy(src.begin(), src.end(), dst.begin());
    return;
  }
  size_t out_index = 0;
  for (size_t i = 0; i < src.size(); ++i) {
    auto mask = masks[i];
    if (std::all_of(mask.begin(), mask.end(), [](uint8_t b) noexcept { return b == 0; })) {
      continue;
    }
    SXT_DEBUG_ASSERT(out_index < dst.size());
    dst[out_index++] = src[i];
  }
}
} // namespace sxt::mtxb
