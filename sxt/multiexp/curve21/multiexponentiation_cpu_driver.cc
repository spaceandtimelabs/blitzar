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
#include "sxt/multiexp/curve21/multiexponentiation_cpu_driver.h"

#include "sxt/base/container/blob_array.h"
#include "sxt/base/error/assert.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/curve21/multiproduct_solver.h"
#include "sxt/multiexp/curve21/multiproducts_combination.h"

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
multiexponentiation_cpu_driver::multiexponentiation_cpu_driver(
    const multiproduct_solver* solver) noexcept
    : solver_{solver} {}

//--------------------------------------------------------------------------------------------------
// compute_multiproduct
//--------------------------------------------------------------------------------------------------
xena::future<memmg::managed_array<void>> multiexponentiation_cpu_driver::compute_multiproduct(
    mtxi::index_table&& multiproduct_table, basct::span_cvoid generators,
    const basct::blob_array& masks, size_t num_inputs) const noexcept {
  auto res =
      solver_
          ->solve(std::move(multiproduct_table),
                  {static_cast<const c21t::element_p3*>(generators.data()), generators.size()},
                  masks, num_inputs)
          .value();
  return xena::make_ready_future<memmg::managed_array<void>>(std::move(res));
}

//--------------------------------------------------------------------------------------------------
// combine_multiproduct_outputs
//--------------------------------------------------------------------------------------------------
xena::future<memmg::managed_array<void>>
multiexponentiation_cpu_driver::combine_multiproduct_outputs(
    xena::future<memmg::managed_array<void>>&& multiproduct,
    basct::blob_array&& output_digit_or_all) const noexcept {
  SXT_DEBUG_ASSERT(multiproduct.ready());
  auto products = std::move(multiproduct.value().as_array<c21t::element_p3>());
  memmg::managed_array<c21t::element_p3> res(output_digit_or_all.size());
  combine_multiproducts(res, output_digit_or_all, products);
  return xena::make_ready_future<memmg::managed_array<void>>(std::move(res));
}
} // namespace sxt::mtxc21
