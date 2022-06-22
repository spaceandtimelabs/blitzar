#pragma once

#include <cstdint>
#include <algorithm>

namespace sxt::basbt {
//--------------------------------------------------------------------------------------------------
// count_trailing_zeros
//--------------------------------------------------------------------------------------------------
inline int count_trailing_zeros(unsigned long x) noexcept {
  return __builtin_ctzl(x);
}

//--------------------------------------------------------------------------------------------------
// count_leading_zeros
//--------------------------------------------------------------------------------------------------
inline int count_leading_zeros(unsigned long x) noexcept {
  return __builtin_clzl(x);
}

//--------------------------------------------------------------------------------------------------
// count_leading_zeros
//--------------------------------------------------------------------------------------------------
inline size_t count_leading_zeros(const uint8_t* data, size_t n) noexcept {
  size_t count = 0;
  while (n > 0) {
    auto k = std::min(n, sizeof(unsigned long));
    unsigned long x = 0;
    std::copy_n(data + (n - k), k, reinterpret_cast<uint8_t*>(&x));
    if (x > 0) {
      count += count_leading_zeros(x) - 8 * (sizeof(unsigned long) - k);
      return count;
    } else {
      count += k * 8;
    }
    n -= k;
  }
  return count;
}

//--------------------------------------------------------------------------------------------------
// pop_count
//--------------------------------------------------------------------------------------------------
inline int pop_count(long long x) noexcept { return __builtin_popcountll(x); }

inline int pop_count(const uint8_t* data, size_t n) noexcept {
  static constexpr size_t block_size = sizeof(long long);
  int res = 0;
  while (n > 0) {
    long long x = 0;
    auto k = std::min(block_size, n);
    std::copy_n(data, k, reinterpret_cast<char*>(&x));
    res += pop_count(x);
    data += k;
    n -= k;
  }
  return res;
}
} // namespace sxt::basbt
