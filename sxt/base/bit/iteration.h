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

#include "sxt/base/bit/count.h"
#include "sxt/base/error/assert.h"

namespace sxt::basbt {
//--------------------------------------------------------------------------------------------------
// consume_next_bit
//--------------------------------------------------------------------------------------------------
inline int consume_next_bit(uint64_t& bitset) noexcept {
  SXT_DEBUG_ASSERT(bitset != 0);
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
