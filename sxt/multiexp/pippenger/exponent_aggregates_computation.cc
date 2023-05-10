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
#include "sxt/multiexp/pippenger/exponent_aggregates_computation.h"

#include <algorithm>

#include "sxt/base/bit/count.h"
#include "sxt/base/bit/span_op.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/pippenger/exponent_aggregates.h"

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// compute_exponent_aggregates
//--------------------------------------------------------------------------------------------------
void compute_exponent_aggregates(exponent_aggregates& aggregates,
                                 basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  size_t max_sequence_length = 0;
  uint8_t max_element_nbytes = 0;
  for (auto& sequence : exponents) {
    max_sequence_length = std::max(max_sequence_length, sequence.n);
    max_element_nbytes = std::max(max_element_nbytes, sequence.element_nbytes);
  }
  aggregates.max_exponent.resize(max_element_nbytes);
  aggregates.term_or_all.resize(max_sequence_length, max_element_nbytes);
  aggregates.output_or_all.resize(exponents.size(), max_element_nbytes);
  aggregates.pop_count = 0;

  for (size_t output_index = 0; output_index < exponents.size(); ++output_index) {
    auto& sequence = exponents[output_index];
    auto element_nbytes = sequence.element_nbytes;
    for (size_t term_index = 0; term_index < sequence.n; ++term_index) {
      basct::cspan<uint8_t> term{sequence.data + term_index * element_nbytes, element_nbytes};
      basbt::or_equal(aggregates.term_or_all[term_index], term);
      basbt::or_equal(aggregates.output_or_all[output_index], term);
      basbt::max_equal(aggregates.max_exponent, term);
      aggregates.pop_count += basbt::pop_count(term);
    }
  }
}
} // namespace sxt::mtxpi
