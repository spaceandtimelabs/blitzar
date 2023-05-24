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
#include "sxt/multiexp/pippenger_multiprod/active_count.h"

#include "sxt/base/error/assert.h"

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// count_active_entries
//--------------------------------------------------------------------------------------------------
void count_active_entries(basct::span<size_t> counts,
                          basct::cspan<basct::cspan<uint64_t>> rows) noexcept {
  for (auto row : rows) {
    SXT_DEBUG_ASSERT(row.size() >= 2 && row.size() >= 2 + row[1]);
    for (size_t active_index = 2 + row[1]; active_index < row.size(); ++active_index) {
      ++counts[row[active_index]];
    }
  }
}
} // namespace sxt::mtxpmp
