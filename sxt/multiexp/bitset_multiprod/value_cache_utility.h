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

#include <cstddef>

#include "sxt/base/error/assert.h"
#include "sxt/multiexp/bitset_multiprod/value_cache.h"

namespace sxt::mtxbmp {
//--------------------------------------------------------------------------------------------------
// compute_cache_size
//--------------------------------------------------------------------------------------------------
size_t compute_cache_size(size_t num_terms) noexcept;

//--------------------------------------------------------------------------------------------------
// init_value_cache_side
//--------------------------------------------------------------------------------------------------
template <class T, class Op>
void init_value_cache_side(basct::span<T> cache_side, Op op, basct::cspan<T> terms) noexcept {
  for (auto& value : cache_side) {
    op.mark_unset(value);
  }
  for (size_t term_index = 0; term_index < terms.size(); ++term_index) {
    cache_side[(1 << term_index) - 1] = terms[term_index];
  }
}

//--------------------------------------------------------------------------------------------------
// init_value_cache
//--------------------------------------------------------------------------------------------------
template <class T, class Op>
void init_value_cache(value_cache<T> cache, Op op, basct::cspan<T> terms) noexcept {
  SXT_DEBUG_ASSERT(terms.size() == cache.num_terms());
  auto [left_cache, right_cache] = cache.split();
  auto left_num_terms = cache.half_num_terms();
  init_value_cache_side(left_cache, op, terms.subspan(0, left_num_terms));
  init_value_cache_side(right_cache, op, terms.subspan(left_num_terms));
}
} // namespace sxt::mtxbmp
