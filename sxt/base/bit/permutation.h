/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
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

#include "sxt/base/bit/count.h"

namespace sxt::basbt {
//--------------------------------------------------------------------------------------------------
// next_permutation
//--------------------------------------------------------------------------------------------------
/**
 * Iterate over all permutations of a number with a given number of bits set to 1.
 *
 * Adopted from https://stackoverflow.com/a/8281965
 */
template <std::unsigned_integral T> T next_permutation(T x) noexcept {
  static constexpr T one{1};
  T t = x | (x - one);
  return (t + one) | (((~t & -~t) - one) >> (basbt::count_trailing_zeros(x) + 1));
}
} // namespace sxt::basbt
