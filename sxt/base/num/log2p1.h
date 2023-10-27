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

#include <concepts>
#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::basn {
//--------------------------------------------------------------------------------------------------
// log2p1
//--------------------------------------------------------------------------------------------------
/**
 * - x represents an unsigned number in little-endian format
 * - x can have a maximum of 1016 bits
 * - x.size() represents the total amount of bytes for the number
 */
double log2p1(basct::cspan<uint8_t> x) noexcept;

//--------------------------------------------------------------------------------------------------
// log2p1
//--------------------------------------------------------------------------------------------------
template <std::integral T> consteval T log2p1(T x) noexcept {
  T res = 0;
  T y = 1;
  while (y <= x) {
    y *= 2;
    res += 1;
  }
  return res;
}
} // namespace sxt::basn
