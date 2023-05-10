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
#include "sxt/multiexp/index/clump2_marker_utility.h"

#include <cmath>

#include "sxt/base/error/assert.h"
#include "sxt/multiexp/index/clump2_descriptor.h"

namespace sxt::mtxi {
//--------------------------------------------------------------------------------------------------
// compute_clump2_marker
//--------------------------------------------------------------------------------------------------
uint64_t compute_clump2_marker(const clump2_descriptor& descriptor, uint64_t clump_index,
                               uint64_t index1, uint64_t index2) noexcept {
  SXT_DEBUG_ASSERT(index1 <= index2);
  auto part1 = clump_index * descriptor.subset_count;
  auto part2 = descriptor.size * index1 - index1 * (index1 - 1) / 2 + (index2 - index1);
  return part1 + part2;
}

uint64_t compute_clump2_marker(const clump2_descriptor& descriptor, uint64_t clump_index,
                               uint64_t index) noexcept {
  return compute_clump2_marker(descriptor, clump_index, index, index);
}

//--------------------------------------------------------------------------------------------------
// unpack_clump2_marker
//--------------------------------------------------------------------------------------------------
void unpack_clump2_marker(uint64_t& clump_index, uint64_t& index1, uint64_t& index2,
                          const clump2_descriptor& descriptor, uint64_t marker) noexcept {
  clump_index = marker / descriptor.subset_count;
  marker -= clump_index * descriptor.subset_count;

  // apply quadratic formula
  //
  // Note: we make use of the property from IEEE 754 that sqrt is "correctly rounded".
  // See https://stackoverflow.com/a/48791460/4447365
  auto b2 = 2 * descriptor.size + 1;
  auto x = std::sqrt(4 * descriptor.size * descriptor.size + 4 * descriptor.size + 1 - 8 * marker);
  index1 = static_cast<uint64_t>(b2 - x) / 2;
  auto u = descriptor.size * index1 - index1 * (index1 - 1) / 2;
  index2 = marker - u + index1;
}

//--------------------------------------------------------------------------------------------------
// consume_clump2_marker
//--------------------------------------------------------------------------------------------------
uint64_t consume_clump2_marker(basct::span<uint64_t>& indexes,
                               const clump2_descriptor& descriptor) noexcept {
  SXT_DEBUG_ASSERT(!indexes.empty());
  auto idx = indexes[0];
  auto clump_index = idx / descriptor.size;
  auto clump_first = clump_index * descriptor.size;
  auto index1 = idx - clump_first;
  auto index2 = descriptor.size;
  if (indexes.size() > 1) {
    SXT_DEBUG_ASSERT(indexes[1] > idx);
    index2 = indexes[1] - clump_first;
  }
  if (index2 >= descriptor.size) {
    indexes = {indexes.data() + 1, indexes.size() - 1};
    return compute_clump2_marker(descriptor, clump_index, index1, index1);
  }
  indexes = {indexes.data() + 2, indexes.size() - 2};
  return compute_clump2_marker(descriptor, clump_index, index1, index2);

  return 0;
}
} // namespace sxt::mtxi
