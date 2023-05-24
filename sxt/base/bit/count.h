/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <algorithm>
#include <cstdint>

namespace sxt::basbt {
//--------------------------------------------------------------------------------------------------
// count_trailing_zeros
//--------------------------------------------------------------------------------------------------
inline int count_trailing_zeros(unsigned long x) noexcept { return __builtin_ctzl(x); }

inline int count_trailing_zeros(unsigned long long x) noexcept { return __builtin_ctzll(x); }

//--------------------------------------------------------------------------------------------------
// count_leading_zeros
//--------------------------------------------------------------------------------------------------
inline int count_leading_zeros(unsigned long x) noexcept { return __builtin_clzl(x); }

inline int count_leading_zeros(unsigned long long x) noexcept { return __builtin_clzll(x); }

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
