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
#pragma once

#include <compare>
#include <cstdint>

#include "sxt/execution/kernel/block_size.h"

namespace sxt::mtxmpg {
//--------------------------------------------------------------------------------------------------
// block_computation_descriptor
//--------------------------------------------------------------------------------------------------
struct block_computation_descriptor {
  unsigned block_offset;
  unsigned index_first;
  unsigned n;
  unsigned reduction_num_blocks;
  xenk::block_size_t block_size;

  auto operator<=>(const block_computation_descriptor&) const noexcept = default;
};
} // namespace sxt::mtxmpg
