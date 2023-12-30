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

  // Note: There's probably a better way to do this that avoids branching, but
  // this is an ok place to start from.
  if (x < 0) {
    return -x;
  } else {
    return x;
  }
}
} // namespace sxt::basn
