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
#include "sxt/multiexp/random/random_multiproduct_generation.h"

#include <cstdlib>
#include <iostream>
#include <vector>

#include "sxt/base/error/panic.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/index/reindex.h"
#include "sxt/multiexp/random/random_multiproduct_descriptor.h"

namespace sxt::mtxrn {
//--------------------------------------------------------------------------------------------------
// generate_random_multiproduct
//--------------------------------------------------------------------------------------------------
void generate_random_multiproduct(mtxi::index_table& products, size_t& num_inputs,
                                  size_t& num_entries, std::mt19937& rng,
                                  const random_multiproduct_descriptor& descriptor) noexcept {
  if (descriptor.max_sequence_length > descriptor.max_num_inputs) {
    baser::panic("max_sequence_length must be less than or equal to max_num_inputs");
  }

  // determine the product table dimensions
  std::uniform_int_distribution<size_t> num_sequences_dist{descriptor.min_num_sequences,
                                                           descriptor.max_num_sequences};
  auto num_sequences = num_sequences_dist(rng);

  std::vector<size_t> sequence_lengths(num_sequences);
  std::uniform_int_distribution<size_t> sequence_length_dist{
      descriptor.min_sequence_length,
      descriptor.max_sequence_length,
  };
  num_entries = 0;
  for (size_t sequence_index = 0; sequence_index < num_sequences; ++sequence_index) {
    auto n = sequence_length_dist(rng);
    sequence_lengths[sequence_index] = n;
    num_entries += n;
  }
  products.reshape(num_sequences, num_entries);

  // fill the product table
  auto entry_data = products.entry_data();
  for (size_t row_index = 0; row_index < products.num_rows(); ++row_index) {
    auto n = sequence_lengths[row_index];
    auto& row = products.header()[row_index];
    row = {entry_data, n};
    uint64_t lower_bound = 0;
    for (size_t index = 0; index < n; ++index) {
      std::uniform_int_distribution<uint64_t> entry_dist{lower_bound,
                                                         descriptor.max_num_inputs - n + index};
      auto entry = entry_dist(rng);
      *entry_data++ = entry;
      lower_bound = entry + 1;
    }
  }

  // reindex the product table
  std::vector<uint64_t> values_data(num_entries);
  basct::span<uint64_t> values = values_data;
  mtxi::reindex_rows(products.header(), values);
  num_inputs = values.size();
}
} // namespace sxt::mtxrn
