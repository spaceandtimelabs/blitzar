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

#include <cmath>
#include <concepts>
#include <type_traits>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::basn {
//--------------------------------------------------------------------------------------------------
// abs
//--------------------------------------------------------------------------------------------------
/**
 * Support abs for integral types larger than 128 bits.
 */
template <std::signed_integral T> CUDA_CALLABLE T abs(T x) noexcept {
  if constexpr (sizeof(T) <= 8) {
    return std::abs(x);
  }
  auto mul = static_cast<int>(x > 0) * 2 - 1;
  return mul * x;
}

//--------------------------------------------------------------------------------------------------
// abs_to_unsigned
//--------------------------------------------------------------------------------------------------
template <std::signed_integral T> CUDA_CALLABLE auto abs_to_unsigned(T x) noexcept {
  using Tp = std::make_unsigned_t<T>;
  // Use some arithmetic to make sure that conversion doesn't overflow
  // for std::numeric_limits<T>::min() since 
  //      -std::numeric_limits<T>::min() == std::numeric_limits<T>::max() + 1
  auto m = static_cast<int>(x > 0) * 2 - 1;
  return static_cast<Tp>(m * (x - m)) + 1;
}
} // namespace sxt::basn
