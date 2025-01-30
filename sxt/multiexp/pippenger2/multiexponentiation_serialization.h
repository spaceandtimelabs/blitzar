/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2025-present Space and Time Labs, Inc.
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

#include <algorithm>
#include <cstddef>
#include <format>
#include <fstream>
#include <memory>
#include <vector>

#include "sxt/base/container/span.h"
#include "sxt/base/system/file_io.h"
#include "sxt/base/curve/element.h"
#include "sxt/multiexp/pippenger2/partition_table_accessor.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// variable_length_multiexponentiation_descriptor
//--------------------------------------------------------------------------------------------------
template <bascrv::element T, class U> struct variable_length_multiexponentiation_descriptor {
  std::unique_ptr<partition_table_accessor<U>> accessor;
  std::vector<unsigned> output_bit_table;
  std::vector<unsigned> output_lengths;
  std::vector<uint8_t> scalars;
};

//--------------------------------------------------------------------------------------------------
// write_multiexponentiation
//--------------------------------------------------------------------------------------------------
template <bascrv::element T, class U>
void write_multiexponentiation(const char* dir, const partition_table_accessor<U>& accessor,
                               basct::cspan<unsigned> output_bit_table,
                               basct::cspan<unsigned> output_lengths,
                               basct::cspan<uint8_t> scalars) noexcept {
  size_t n = 0;
  if (!output_lengths.empty()) {
     n = *std::max_element(output_lengths.begin(), output_lengths.end());
  }
  bassy::write_file(std::format("{}/output_bit_table.bin", dir), output_bit_table);
  bassy::write_file(std::format("{}/output_lengths.bin", dir), output_lengths);
  bassy::write_file(std::format("{}/scalars.bin", dir), scalars);

  std::vector<T> generators(n);
  accessor.copy_generators(generators);
  bassy::write_file(std::format("{}/generators.bin", dir), generators);
}
} // namespace sxt::mtxpp2
