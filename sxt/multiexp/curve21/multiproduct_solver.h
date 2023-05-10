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

#include "sxt/base/container/span.h"
#include "sxt/execution/async/future_fwd.h"
#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::basct {
class blob_array;
}
namespace sxt::mtxi {
class index_table;
}
namespace sxt::c21t {
struct element_p3;
}

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// multiproduct_solver
//--------------------------------------------------------------------------------------------------
class multiproduct_solver {
public:
  virtual ~multiproduct_solver() noexcept = default;

  virtual xena::future<memmg::managed_array<c21t::element_p3>>
  solve(mtxi::index_table&& multiproduct_table, basct::cspan<c21t::element_p3> generators,
        const basct::blob_array& mask, size_t num_inputs) const noexcept = 0;
};
} // namespace sxt::mtxc21
