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
#include "sxt/multiexp/pippenger2/variable_length_computation.h"

#include <algorithm>

#include "sxt/base/error/assert.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// compute_product_length_table
//--------------------------------------------------------------------------------------------------
void compute_product_length_table(basct::span<unsigned>& product_lengths,
                                  basct::cspan<unsigned> bit_widths,
                                  basct::cspan<unsigned> output_lengths, unsigned first,
                                  unsigned length) noexcept {
  auto num_products = product_lengths.size();
  auto num_outputs = bit_widths.size();
  SXT_DEBUG_ASSERT(
      // clang-format off
      num_products >= num_outputs &&
      product_lengths.size() == num_products &&
      bit_widths.size() == num_outputs &&
      output_lengths.size() == num_outputs
      // clang-format on
  );

  // find first output with longer than <length>
  auto output_first =
      std::count_if(output_lengths.begin(), output_lengths.end(),
                    [&](double output_length) noexcept { return output_length <= first; });

  // fill in product lengths
  unsigned product_index = 0;
  for (auto output_index = output_first; output_index < num_outputs; ++output_index) {
    auto output_length = output_lengths[output_index];
    SXT_DEBUG_ASSERT(output_length > first);
    auto product_length = std::min(output_length - first, length);
    auto bit_width = bit_widths[output_index];
    for (unsigned bit_index = 0; bit_index < bit_width; ++bit_index) {
      product_lengths[product_index++] = product_length;
    }
  }
  product_lengths = product_lengths.subspan(0, product_index);
}
} // namespace sxt::mtxpp2
