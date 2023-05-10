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
#include "sxt/multiexp/index/partition_marker_utility.h"

#include "sxt/base/bit/count.h"
#include "sxt/base/error/assert.h"

namespace sxt::mtxi {
//--------------------------------------------------------------------------------------------------
// consume_partition_marker
//--------------------------------------------------------------------------------------------------
uint64_t consume_partition_marker(basct::span<uint64_t>& indexes,
                                  uint64_t partition_size) noexcept {
  SXT_DEBUG_ASSERT(!indexes.empty());
  SXT_DEBUG_ASSERT(partition_size > 0 && partition_size < 64);
  auto n = indexes.size();
  auto idx = indexes[0];
  auto partition_index = idx / partition_size;
  auto partition_first = partition_index * partition_size;
  auto partition_offset = idx - partition_index * partition_size;
  uint64_t marker = (partition_index << partition_size) | (1 << partition_offset);
  size_t i = 1;
  for (; i < n; ++i) {
    SXT_DEBUG_ASSERT(idx < indexes[i]);
    idx = indexes[i];
    partition_offset = idx - partition_first;
    if (partition_offset >= partition_size) {
      break;
    }
    marker |= (1 << partition_offset);
  }
  indexes = basct::span{indexes.data() + i, n - i};
  return marker;
}
} // namespace sxt::mtxi
