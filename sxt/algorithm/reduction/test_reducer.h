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

#include <cstdint>

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::algr {
//--------------------------------------------------------------------------------------------------
// test_add_reducer
//--------------------------------------------------------------------------------------------------
struct test_add_reducer {
  using value_type = uint64_t;

  static constexpr uint64_t identity() noexcept {
    return 0;
  }

  template <class T> static inline CUDA_CALLABLE void accumulate(T& res, uint64_t x) noexcept {
    res = res + x;
  }

  template <class T>
  static inline CUDA_CALLABLE void accumulate_inplace(T& res, uint64_t x) noexcept {
    res = res + x;
  }
};
} // namespace sxt::algr
