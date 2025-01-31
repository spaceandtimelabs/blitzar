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
#include <numeric>
#include <vector>

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/base/system/file_io.h"
#include "sxt/multiexp/pippenger2/in_memory_partition_table_accessor_utility.h"
#include "sxt/multiexp/pippenger2/partition_table_accessor.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// packed_multiexponentiation_descriptor
//--------------------------------------------------------------------------------------------------
template <bascrv::element T, class U> struct packed_multiexponentiation_descriptor {
  std::unique_ptr<partition_table_accessor<U>> accessor;
  std::vector<unsigned> output_bit_table;
  std::vector<uint8_t> scalars;
};

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
void write_multiexponentiation(std::string_view dir, const partition_table_accessor<U>& accessor,
                               basct::cspan<unsigned> output_bit_table,
                               basct::cspan<uint8_t> scalars) noexcept {
  size_t num_products = std::accumulate(output_bit_table.begin(), output_bit_table.end(), 0ull);
  auto num_output_bytes = basn::divide_up<size_t>(num_products, 8);
  SXT_DEBUG_ASSERT(
      scalars.size() % num_output_bytes == 0
  );
  auto n = scalars.size() / num_output_bytes;

  bassy::write_file(std::format("{}/output_bit_table.bin", dir), output_bit_table);
  bassy::write_file(std::format("{}/scalars.bin", dir), scalars);

  std::vector<U> generators(n);
  accessor.copy_generators(generators);
  bassy::write_file<U>(std::format("{}/generators.bin", dir), generators);

  uint64_t window_width = accessor.window_width();
  bassy::write_file(std::format("{}/window_width.bin", dir),
                    basct::cspan<uint64_t>{&window_width, 1});
}

template <bascrv::element T, class U>
void write_multiexponentiation(std::string_view dir, const partition_table_accessor<U>& accessor,
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

  std::vector<U> generators(n);
  accessor.copy_generators(generators);
  bassy::write_file<U>(std::format("{}/generators.bin", dir), generators);

  uint64_t window_width = accessor.window_width();
  bassy::write_file(std::format("{}/window_width.bin", dir),
                    basct::cspan<uint64_t>{&window_width, 1});
}

//--------------------------------------------------------------------------------------------------
// read_multiexponentiation
//--------------------------------------------------------------------------------------------------
template <bascrv::element T, class U>
void read_multiexponentiation(packed_multiexponentiation_descriptor<T, U>& descr,
                              std::string_view dir) noexcept {
  bassy::read_file(descr.output_bit_table, std::format("{}/output_bit_table.bin", dir));
  bassy::read_file(descr.scalars, std::format("{}/scalars.bin", dir));

  // accessor
  std::vector<uint64_t> window_width;
  bassy::read_file(window_width, std::format("{}/window_width.bin", dir));
  SXT_RELEASE_ASSERT(window_width.size() == 1);
  std::vector<U> generators;
  bassy::read_file(generators, std::format("{}/generators.bin", dir));
  std::vector<T> generators_p{generators.begin(), generators.end()};
  descr.accessor =
      make_in_memory_partition_table_accessor<U>(generators_p, basm::alloc_t{}, window_width[0]);
}

template <bascrv::element T, class U>
void read_multiexponentiation(variable_length_multiexponentiation_descriptor<T, U>& descr,
                              std::string_view dir) noexcept {
  bassy::read_file(descr.output_bit_table, std::format("{}/output_bit_table.bin", dir));
  bassy::read_file(descr.output_lengths, std::format("{}/output_lengths.bin", dir));
  bassy::read_file(descr.scalars, std::format("{}/scalars.bin", dir));

  // accessor
  std::vector<uint64_t> window_width;
  bassy::read_file(window_width, std::format("{}/window_width.bin", dir));
  SXT_RELEASE_ASSERT(window_width.size() == 1);
  std::vector<U> generators;
  bassy::read_file(generators, std::format("{}/generators.bin", dir));
  std::vector<T> generators_p{generators.begin(), generators.end()};
  descr.accessor =
      make_in_memory_partition_table_accessor<U>(generators_p, basm::alloc_t{}, window_width[0]);
}
} // namespace sxt::mtxpp2
