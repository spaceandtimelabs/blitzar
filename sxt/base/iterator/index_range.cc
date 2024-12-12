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
#include "sxt/base/iterator/index_range.h"

#include "sxt/base/error/assert.h"

namespace sxt::basit {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
index_range::index_range(size_t a, size_t b) noexcept : index_range{a, b, 1} {}

index_range::index_range(size_t a, size_t b, size_t chunk_multiple) noexcept
    : a_{a}, b_{b}, chunk_multiple_{chunk_multiple} {
  SXT_DEBUG_ASSERT(
      // clang-format off
      0 <= a && a <= b &&
      chunk_multiple > 0
      // clang-format on
  );
}

//--------------------------------------------------------------------------------------------------
// chunk_multiple
//--------------------------------------------------------------------------------------------------
index_range index_range::chunk_multiple(size_t val) const noexcept {
  return {
      a_,
      b_,
      val,
  };
}
} // namespace sxt::basit
