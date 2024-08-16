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
#include "sxt/cbindings/backend/computational_backend_utility.h"

#include <algorithm>

#include "sxt/base/error/assert.h"
#include "sxt/base/num/divide_up.h"

namespace sxt::cbnbck {
//--------------------------------------------------------------------------------------------------
// make_scalars_span
//--------------------------------------------------------------------------------------------------
basct::cspan<uint8_t> make_scalars_span(const uint8_t* data,
                                        basct::cspan<unsigned> output_bit_table,
                                        basct::cspan<unsigned> output_lengths) noexcept {
  auto num_outputs = output_bit_table.size();
  SXT_DEBUG_ASSERT(output_lengths.size() == num_outputs);

  unsigned output_bit_sum = 0;
  unsigned n = 0;
  unsigned prev_len = 0;
  for (unsigned output_index = 0; output_index < num_outputs; ++output_index) {
    auto width = output_bit_table[output_index];
    SXT_RELEASE_ASSERT(width > 0, "output bit width must be positive");
    auto len = output_lengths[output_index];
    SXT_RELEASE_ASSERT(len >= prev_len, "output lengths must be sorted in ascending order");

    output_bit_sum += width;
    n = std::max(n, len);
    prev_len = len;
  }

  auto output_num_bytes = basn::divide_up(output_bit_sum, 8u);
  return basct::cspan<uint8_t>{data, output_num_bytes * n};
}
} // namespace sxt::cbnbck
