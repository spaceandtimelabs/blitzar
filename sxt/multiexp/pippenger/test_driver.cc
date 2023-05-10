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
#include "sxt/multiexp/pippenger/test_driver.h"

#include "sxt/base/bit/span_op.h"
#include "sxt/base/container/blob_array.h"
#include "sxt/base/container/span.h"
#include "sxt/base/container/span_void.h"
#include "sxt/base/error/assert.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/base/generator_utility.h"
#include "sxt/multiexp/index/index_table.h"

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// compute_multiproduct
//--------------------------------------------------------------------------------------------------
xena::future<memmg::managed_array<void>>
test_driver::compute_multiproduct(mtxi::index_table&& multiproduct_table,
                                  basct::span_cvoid generators, const basct::blob_array& masks,
                                  size_t num_inputs) const noexcept {
  memmg::managed_array<uint64_t> inputs_data;
  basct::cspan<uint64_t> inputs;
  auto generators_p =
      basct::cspan<uint64_t>{static_cast<const uint64_t*>(generators.data()), generators.size()};
  if (num_inputs == generators.size()) {
    inputs = generators_p;
  } else {
    inputs_data = memmg::managed_array<uint64_t>(num_inputs);
    mtxb::filter_generators<uint64_t>(inputs_data, generators_p, masks);
    inputs = inputs_data;
  }
  memmg::managed_array<uint64_t> outputs(multiproduct_table.num_rows());
  for (size_t row_index = 0; row_index < multiproduct_table.num_rows(); ++row_index) {
    auto products = multiproduct_table.header()[row_index];
    SXT_RELEASE_ASSERT(!products.empty());
    auto& output = outputs[row_index];
    output = 0;
    for (size_t product_index = 2; product_index < products.size(); ++product_index) {
      auto input = products[product_index];
      SXT_RELEASE_ASSERT(input < inputs.size());
      output += inputs[input];
    }
  }
  return xena::make_ready_future(std::move(static_cast<memmg::managed_array<void>&>(outputs)));
}

//--------------------------------------------------------------------------------------------------
// combine_multiproduct_outputs
//--------------------------------------------------------------------------------------------------
xena::future<memmg::managed_array<void>>
test_driver::combine_multiproduct_outputs(xena::future<memmg::managed_array<void>>&& multiproduct,
                                          basct::blob_array&& output_digit_or_all) const noexcept {
  auto& multiproduct_array = multiproduct.value();
  basct::cspan<uint64_t> inputs{static_cast<uint64_t*>(multiproduct_array.data()),
                                multiproduct_array.size()};
  memmg::managed_array<uint64_t> outputs(output_digit_or_all.size());
  size_t input_index = 0;
  for (size_t output_index = 0; output_index < output_digit_or_all.size(); ++output_index) {
    auto& output = outputs[output_index];
    output = 0;
    basbt::for_each_bit(output_digit_or_all[output_index], [&](size_t pos) noexcept {
      SXT_DEBUG_ASSERT(input_index < inputs.size());
      output += (1ull << pos) * inputs[input_index++];
    });
  }
  return xena::make_ready_future(std::move(static_cast<memmg::managed_array<void>&>(outputs)));
}
} // namespace sxt::mtxpi
