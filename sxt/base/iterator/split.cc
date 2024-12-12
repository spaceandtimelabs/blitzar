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
#include "sxt/base/iterator/split.h"

#include <algorithm>

#include "sxt/base/error/assert.h"
#include "sxt/base/num/divide_up.h"

namespace sxt::basit {
//--------------------------------------------------------------------------------------------------
// split
//--------------------------------------------------------------------------------------------------
std::pair<index_range_iterator, index_range_iterator> split(const index_range& rng,
                                                            const split_options& options) noexcept {
  SXT_DEBUG_ASSERT(options.split_factor > 0);
  auto delta = rng.b() - rng.a();
  auto step = std::max(basn::divide_up(delta, options.split_factor), size_t{1});
  step = std::max(step, options.min_chunk_size);
  step = std::min(step, options.max_chunk_size);
  step = basn::divide_up(step, rng.chunk_multiple()) * rng.chunk_multiple();
  index_range_iterator first{index_range{rng.a(), rng.b()}, step};
  index_range_iterator last{index_range{rng.b(), rng.b()}, step};
  return {first, last};
}
} // namespace sxt::basit
