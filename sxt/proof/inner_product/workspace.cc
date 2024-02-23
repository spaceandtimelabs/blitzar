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
#include "sxt/proof/inner_product/workspace.h"

#include "sxt/base/container/span_utility.h"
#include "sxt/proof/inner_product/proof_descriptor.h"

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
workspace::workspace(std::pmr::memory_resource* upstream) noexcept : alloc{upstream} {}

//--------------------------------------------------------------------------------------------------
// init_workspace
//--------------------------------------------------------------------------------------------------
void init_workspace(workspace& work) noexcept {
  auto np_half = work.descriptor->g_vector.size() / 2u;

  work.round_index = 0;

  auto scalars = basct::winked_span<s25t::element>(&work.alloc, 2u * np_half);

  // a_vector
  work.a_vector = scalars.subspan(0, np_half);

  // b_vector
  work.b_vector = scalars.subspan(np_half);

  // g_vector
  work.g_vector = basct::winked_span<c32t::element_p3>(&work.alloc, np_half);
}
} // namespace sxt::prfip
