#pragma once

#include <cstddef>

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"

namespace sxt::mtxbk {
//--------------------------------------------------------------------------------------------------
// accumulate_buckets
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
void accumulate_buckets(basct::span<T> bucket_sums, basct::cspan<T> generators,
                        basct::cspan<const uint8_t*> exponents) noexcept {
  (void)bucket_sums;
  (void)generators;
  (void)exponents;
}
} // namespace sxt::mtxbk
