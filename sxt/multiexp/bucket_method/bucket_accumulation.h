#pragma once

#include <cstddef>

#include "sxt/base/error/assert.h"
#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// accumulate_buckets
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void accumulate_buckets(basct::span<T> bucket_sums, basct::cspan<T> generators,
                        basct::cspan<const uint8_t*> exponents) noexcept {
  constexpr size_t bucket_group_size = 255;
  constexpr size_t num_bucket_groups = 32;
  auto num_outputs = exponents.size();
  SXT_DEBUG_ASSERT(
      bucket_sums.size() == bucket_group_size * num_bucket_groups * num_outputs
  );
  for (auto& e : bucket_sums) {
    e = T::identity();
  }
  (void)bucket_sums;
  (void)generators;
  (void)exponents;
}
} // namespace sxt::mtxbk
