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
#include "sxt/multiexp/pippenger_multiprod/multiproduct_params_computation.h"

#include <cmath>

#include "sxt/multiexp/pippenger_multiprod/multiproduct_params.h"

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// compute_multiproduct_params
//--------------------------------------------------------------------------------------------------
void compute_multiproduct_params(multiproduct_params& params, size_t num_outputs,
                                 size_t num_inputs) noexcept {
  if (num_inputs == 0) {
    params = {};
    return;
  }
  params.partition_size = static_cast<size_t>(std::ceil(std::log2(num_outputs * num_inputs)));
  if (params.partition_size <= 3 || params.partition_size >= num_inputs / 2) {
    params.partition_size = 0;
  }
  auto clump_size = std::ceil(num_inputs / std::log2(num_inputs + 1));
  params.input_clump_size = clump_size;
  params.output_clump_size = clump_size;
}
} // namespace sxt::mtxpmp
