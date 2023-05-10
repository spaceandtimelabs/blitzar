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
#include "sxt/multiexp/bitset_multiprod/value_cache_utility.h"

namespace sxt::mtxbmp {
//--------------------------------------------------------------------------------------------------
// compute_cache_size
//--------------------------------------------------------------------------------------------------
size_t compute_cache_size(size_t num_terms) noexcept {
  auto left_size = num_terms / 2;
  auto right_size = num_terms - left_size;
  return (1 << left_size) + (1 << right_size) - 2;
}
} // namespace sxt::mtxbmp
