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

#include "sxt/base/container/blob_array.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/curve/multiproduct_solver.h"
#include "sxt/multiexp/curve/multiproducts_combination.h"
#include "sxt/multiexp/pippenger/driver.h"

namespace sxt::mtxcrv {
//--------------------------------------------------------------------------------------------------
// multiexponentiation_cpu_driver
//--------------------------------------------------------------------------------------------------
template <bascrv::element Element>
class multiexponentiation_cpu_driver final : public mtxpi::driver {
public:
  //--------------------------------------------------------------------------------------------------
  // constructor
  //--------------------------------------------------------------------------------------------------
  explicit multiexponentiation_cpu_driver(const multiproduct_solver<Element>* solver) noexcept
      : solver_{solver} {}

  //--------------------------------------------------------------------------------------------------
  // compute_multiproduct
  //--------------------------------------------------------------------------------------------------
  xena::future<memmg::managed_array<void>>
  compute_multiproduct(mtxi::index_table&& multiproduct_table, basct::span_cvoid generators,
                       const basct::blob_array& masks, size_t num_inputs) const noexcept override {
    auto res = solver_
                   ->solve(std::move(multiproduct_table),
                           {static_cast<const Element*>(generators.data()), generators.size()},
                           masks, num_inputs)
                   .value();
    return xena::make_ready_future<memmg::managed_array<void>>(std::move(res));
  }

  //--------------------------------------------------------------------------------------------------
  // combine_multiproduct_outputs
  //--------------------------------------------------------------------------------------------------
  xena::future<memmg::managed_array<void>> combine_multiproduct_outputs(
      xena::future<memmg::managed_array<void>>&& multiproduct,
      basct::blob_array&& output_digit_or_all,
      basct::cspan<mtxb::exponent_sequence> exponents) const noexcept override {
    SXT_DEBUG_ASSERT(multiproduct.ready());
    auto products = std::move(multiproduct.value().as_array<Element>());
    memmg::managed_array<Element> res(exponents.size());
    combine_multiproducts<Element>(res, output_digit_or_all, products, exponents);
    return xena::make_ready_future<memmg::managed_array<void>>(std::move(res));
  }

private:
  const multiproduct_solver<Element>* solver_;
};
} // namespace sxt::mtxcrv
