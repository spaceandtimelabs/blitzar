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

#include "sxt/execution/kernel/block_size.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/multiproduct_gpu/block_computation_descriptor.h"

namespace sxt::mtxmpg {
//--------------------------------------------------------------------------------------------------
// multiproduct_computation_descriptor
//--------------------------------------------------------------------------------------------------
struct multiproduct_computation_descriptor {
  unsigned num_blocks;
  xenk::block_size_t max_block_size;
  memmg::managed_array<block_computation_descriptor> block_descriptors;

  bool operator==(const multiproduct_computation_descriptor&) const noexcept = default;
};
} // namespace sxt::mtxmpg
