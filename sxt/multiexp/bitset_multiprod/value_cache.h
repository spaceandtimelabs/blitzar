/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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

#include <cstddef>
#include <utility>

#include "sxt/base/container/span.h"

namespace sxt::mtxbmp {
//--------------------------------------------------------------------------------------------------
// value_cache
//--------------------------------------------------------------------------------------------------
template <class T> class value_cache {
public:
  value_cache() noexcept = default;

  value_cache(T* data, size_t num_terms) noexcept : data_{data}, num_terms_{num_terms} {}

  // accessors
  size_t num_terms() const noexcept { return num_terms_; }

  size_t half_num_terms() const noexcept { return num_terms_ / 2; }

  std::pair<basct::span<T>, basct::span<T>> split() const noexcept {
    auto left_num_terms = this->half_num_terms();
    auto left_cache_size = (1ull << left_num_terms) - 1;
    auto right_num_terms = num_terms_ - left_num_terms;
    auto right_cache_size = (1ull << right_num_terms) - 1;
    return {
        {data_, left_cache_size},
        {data_ + left_cache_size, right_cache_size},
    };
  }

private:
  T* data_{nullptr};
  size_t num_terms_{0};
};
} // namespace sxt::mtxbmp
