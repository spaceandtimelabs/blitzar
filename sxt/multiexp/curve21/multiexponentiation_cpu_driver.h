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

#include "sxt/multiexp/pippenger/driver.h"

namespace sxt::mtxc21 {
class multiproduct_solver;

//--------------------------------------------------------------------------------------------------
// multiexponentiation_cpu_driver
//--------------------------------------------------------------------------------------------------
class multiexponentiation_cpu_driver final : public mtxpi::driver {
public:
  explicit multiexponentiation_cpu_driver(const multiproduct_solver* solver) noexcept;

  // mtxpi::driver
  xena::future<memmg::managed_array<void>>
  compute_multiproduct(mtxi::index_table&& multiproduct_table, basct::span_cvoid generators,
                       const basct::blob_array& masks, size_t num_inputs) const noexcept override;

  xena::future<memmg::managed_array<void>> combine_multiproduct_outputs(
      xena::future<memmg::managed_array<void>>&& multiproduct,
      basct::blob_array&& output_digit_or_all,
      basct::cspan<mtxb::exponent_sequence> exponents) const noexcept override;

private:
  const multiproduct_solver* solver_;
};
} // namespace sxt::mtxc21
