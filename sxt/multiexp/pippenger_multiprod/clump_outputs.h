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
#pragma once

#include <cstdint>
#include <vector>

#include "sxt/base/container/span.h"

namespace sxt::mtxi {
class index_table;
}

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// compute_clumped_output_table
//--------------------------------------------------------------------------------------------------
bool compute_clumped_output_table(mtxi::index_table& table, std::vector<uint64_t>& output_clumps,
                                  basct::cspan<basct::cspan<uint64_t>> rows,
                                  size_t num_active_inputs, size_t clump_size) noexcept;

inline bool compute_clumped_output_table(mtxi::index_table& table,
                                         std::vector<uint64_t>& output_clumps,
                                         basct::span<basct::span<uint64_t>> rows,
                                         size_t num_active_inputs, size_t clump_size) noexcept {
  return compute_clumped_output_table(table, output_clumps,
                                      {
                                          reinterpret_cast<basct::cspan<uint64_t>*>(rows.data()),
                                          rows.size(),
                                      },
                                      num_active_inputs, clump_size);
}

//--------------------------------------------------------------------------------------------------
// rewrite_multiproducts_with_output_clumps
//--------------------------------------------------------------------------------------------------
void rewrite_multiproducts_with_output_clumps(basct::span<basct::span<uint64_t>> rows,
                                              basct::cspan<uint64_t> output_clumps,
                                              size_t clump_size) noexcept;
} // namespace sxt::mtxpmp
