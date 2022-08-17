#pragma once

#include <algorithm>
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

//--------------------------------------------------------------------------------------------------
// for_each_bit
//--------------------------------------------------------------------------------------------------
template <class F> void for_each_bit(const uint8_t* bitset, size_t size, F f) noexcept {
  uint64_t x;
  size_t offset = 0;
  while (size >= sizeof(x)) {
    std::copy_n(bitset, sizeof(x), reinterpret_cast<uint8_t*>(&x));
    bitset += sizeof(x);
    while (x != 0) {
      auto i = consume_next_bit(x);
      f(offset + static_cast<size_t>(i));
    }
    offset += sizeof(x) * 8;
    size -= sizeof(x);
  }
  x = 0;
  std::copy_n(bitset, size, reinterpret_cast<uint8_t*>(&x));
  while (x != 0) {
    auto i = consume_next_bit(x);
    f(offset + static_cast<size_t>(i));
  }
}
} // namespace sxt::basbt
