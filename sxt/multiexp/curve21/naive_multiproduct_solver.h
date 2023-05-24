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

#include "sxt/multiexp/curve21/multiproduct_solver.h"

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// naive_multiproduct_solver
//--------------------------------------------------------------------------------------------------
class naive_multiproduct_solver final : public multiproduct_solver {
public:
  // multiproduct_solver
  xena::future<memmg::managed_array<c21t::element_p3>>
  solve(mtxi::index_table&& multiproduct_table, basct::cspan<c21t::element_p3> generators,
        const basct::blob_array& masks, size_t num_inputs) const noexcept override;
};
} // namespace sxt::mtxc21
