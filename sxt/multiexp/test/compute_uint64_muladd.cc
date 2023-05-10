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
#include "sxt/multiexp/test/compute_uint64_muladd.h"

#include "sxt/multiexp/base/exponent_sequence.h"

namespace sxt::mtxtst {
//--------------------------------------------------------------------------------------------------
// compute_uint64_muladd
//--------------------------------------------------------------------------------------------------
void compute_uint64_muladd(basct::span<uint64_t> result, basct::span<uint64_t> generators,
                           basct::span<mtxb::exponent_sequence> sequences) noexcept {

  // compute the expected result
  for (size_t seq = 0; seq < result.size(); ++seq) {
    result[seq] = 0;

    uint8_t element_nbytes = sequences[seq].element_nbytes;

    // sum all the elements in the current sequence together
    for (size_t gen_i = 0; gen_i < sequences[seq].n; ++gen_i) {
      uint64_t pow256 = 1;
      uint64_t curr_exponent = 0;

      // reconstructs the gen_i-th data element out of its element_nbytes values
      for (size_t j = 0; j < element_nbytes; ++j) {
        curr_exponent += sequences[seq].data[gen_i * element_nbytes + j] * pow256;
        pow256 *= 256;
      }

      result[seq] += generators[gen_i] * curr_exponent;
    }
  }
}
} // namespace sxt::mtxtst
