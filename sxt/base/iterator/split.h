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

#include <limits>
#include <utility>

#include "sxt/base/iterator/index_range.h"
#include "sxt/base/iterator/index_range_iterator.h"

namespace sxt::basit {
//--------------------------------------------------------------------------------------------------
// split_options
//--------------------------------------------------------------------------------------------------
struct split_options {
  size_t min_chunk_size = 1;
  size_t max_chunk_size = std::numeric_limits<size_t>::max();
  size_t split_factor = 1;
};

//--------------------------------------------------------------------------------------------------
// split
//--------------------------------------------------------------------------------------------------
std::pair<index_range_iterator, index_range_iterator> split(const index_range& rng,
                                                            const split_options& options) noexcept;
} // namespace sxt::basit
