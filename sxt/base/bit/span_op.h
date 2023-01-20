#pragma once

#include <algorithm>

#include "sxt/base/bit/count.h"
#include "sxt/base/bit/iteration.h"
#include "sxt/base/container/span.h"
#include "sxt/base/error/assert.h"

namespace sxt::basbt {
//--------------------------------------------------------------------------------------------------
// or_equal
//--------------------------------------------------------------------------------------------------
inline void or_equal(basct::span<uint8_t> lhs, basct::cspan<uint8_t> rhs) noexcept {
  SXT_DEBUG_ASSERT(lhs.size() >= rhs.size());
  for (size_t i = 0; i < rhs.size(); ++i) {
    lhs[i] |= rhs[i];
  }
}

//--------------------------------------------------------------------------------------------------
// max_equal
//--------------------------------------------------------------------------------------------------
inline void max_equal(basct::span<uint8_t> lhs, basct::cspan<uint8_t> rhs) noexcept {
  SXT_DEBUG_ASSERT(lhs.size() >= rhs.size());
  for (auto byte : lhs.subspan(rhs.size())) {
    if (byte != 0) {
      return;
    }
  }
  for (size_t i = rhs.size(); i-- > 0;) {
    if (lhs[i] > rhs[i]) {
      return;
    } else if (lhs[i] < rhs[i]) {
      std::copy_n(rhs.data(), rhs.size(), lhs.data());
      return;
    }
  }
}

//--------------------------------------------------------------------------------------------------
// pop_count
//--------------------------------------------------------------------------------------------------
inline size_t pop_count(basct::cspan<uint8_t> s) noexcept { return pop_count(s.data(), s.size()); }

//--------------------------------------------------------------------------------------------------
// for_each_bit
//--------------------------------------------------------------------------------------------------
template <class F> void for_each_bit(basct::cspan<uint8_t> bitset, F f) noexcept {
  for_each_bit(bitset.data(), bitset.size(), f);
}

//--------------------------------------------------------------------------------------------------
// test_bit
//--------------------------------------------------------------------------------------------------
inline bool test_bit(basct::cspan<uint8_t> bitset, size_t index) noexcept {
  SXT_DEBUG_ASSERT(index < bitset.size() * 8);
  auto byte_pos = index / 8;
  return static_cast<bool>(bitset[byte_pos] & (1 << (index - 8 * byte_pos)));
}

//--------------------------------------------------------------------------------------------------
// count_leading_zeros
//--------------------------------------------------------------------------------------------------
inline size_t count_leading_zeros(basct::cspan<uint8_t> bitset) noexcept {
  return count_leading_zeros(bitset.data(), bitset.size());
}
} // namespace sxt::basbt
