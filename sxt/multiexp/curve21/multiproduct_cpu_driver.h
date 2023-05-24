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

#include "sxt/multiexp/pippenger_multiprod/driver.h"

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// multiproduct_cpu_driver
//--------------------------------------------------------------------------------------------------
class multiproduct_cpu_driver final : public mtxpmp::driver {
public:
  void apply_partition_operation(basct::span_void inout, basct::cspan<uint64_t> partition_markers,
                                 size_t partition_size) const noexcept override;

  void apply_clump2_operation(basct::span_void inout, basct::cspan<uint64_t> markers,
                              const mtxi::clump2_descriptor& descriptor) const noexcept override;

  void compute_naive_multiproduct(basct::span_void inout,
                                  basct::cspan<basct::cspan<uint64_t>> products,
                                  size_t num_inactive_inputs) const noexcept override;

  void permute_inputs(basct::span_void inout,
                      basct::cspan<uint64_t> permutation) const noexcept override;

private:
};
} // namespace sxt::mtxc21
