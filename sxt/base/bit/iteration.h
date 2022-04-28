#pragma once

#include <cassert>
#include <cstdint>

#include "sxt/base/bit/count.h"

namespace sxt::basbt {
//--------------------------------------------------------------------------------------------------
// consume_next_bit
//--------------------------------------------------------------------------------------------------
inline int consume_next_bit(uint64_t& bitset) noexcept {
  assert(bitset != 0);
  // adopted from https://lemire.me/blog/2018/02/21/iterating-over-set-bits-quickly/
  auto res = count_trailing_zeros(bitset);
  bitset ^= bitset & -bitset;
  return res;
}
} // namespace sxt::basbt
