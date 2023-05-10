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
#include "sxt/multiexp/curve21/naive_multiproduct_solver.h"

#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/base/generator_utility.h"
#include "sxt/multiexp/index/index_table.h"

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// solve
//--------------------------------------------------------------------------------------------------
xena::future<memmg::managed_array<c21t::element_p3>>
naive_multiproduct_solver::solve(mtxi::index_table&& multiproduct_table,
                                 basct::cspan<c21t::element_p3> generators,
                                 const basct::blob_array& masks, size_t num_inputs) const noexcept {
  memmg::managed_array<c21t::element_p3> inputs_data;
  basct::cspan<c21t::element_p3> inputs;
  if (num_inputs == generators.size()) {
    inputs = generators;
  } else {
    inputs_data = memmg::managed_array<c21t::element_p3>(num_inputs);
    mtxb::filter_generators<c21t::element_p3>(inputs_data, generators, masks);
    inputs = inputs_data;
  }
  memmg::managed_array<c21t::element_p3> res(multiproduct_table.num_rows());
  for (size_t row_index = 0; row_index < multiproduct_table.num_rows(); ++row_index) {
    auto products = multiproduct_table.header()[row_index];
    SXT_DEBUG_ASSERT(products.size() > 2);
    c21t::element_p3 output = inputs[products[2]];
    for (size_t input_index = 3; input_index < products.size(); ++input_index) {
      auto input = products[input_index];
      c21o::add(output, output, inputs[input]);
    }
    res[row_index] = output;
  }
  return xena::make_ready_future(std::move(res));
}
} // namespace sxt::mtxc21
