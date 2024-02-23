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
#include "sxt/seqcommit/generator/cpu_one_commitments.h"

#include <vector>

#include "sxt/curve32/operation/add.h"
#include "sxt/curve32/type/element_p3.h"
#include "sxt/seqcommit/generator/precomputed_generators.h"

namespace sxt::sqcgn {
//--------------------------------------------------------------------------------------------------
// cpu_get_one_commitments
//--------------------------------------------------------------------------------------------------
void cpu_get_one_commitments(basct::span<c32t::element_p3> one_commitments) noexcept {
  auto prev_commit = c32t::element_p3::identity();

  auto n = one_commitments.size();
  std::vector<c32t::element_p3> generators_data;
  auto precomputed_gens = get_precomputed_generators(generators_data, n, 0, false);

  for (uint64_t i = 0; i < n; ++i) {
    one_commitments[i] = prev_commit;
    c32o::add(prev_commit, prev_commit, precomputed_gens[i]);
  }
}

//--------------------------------------------------------------------------------------------------
// cpu_get_one_commit
//--------------------------------------------------------------------------------------------------
c32t::element_p3 cpu_get_one_commit(c32t::element_p3 prev_commit, uint64_t n,
                                    uint64_t offset) noexcept {
  std::vector<c32t::element_p3> generators_data;
  auto precomputed_gens = get_precomputed_generators(generators_data, n, offset, false);

  for (uint64_t i = 0; i < n; ++i) {
    c32o::add(prev_commit, prev_commit, precomputed_gens[i]);
  }

  return prev_commit;
}

c32t::element_p3 cpu_get_one_commit(uint64_t n) noexcept {
  return cpu_get_one_commit(c32t::element_p3::identity(), n, 0);
}
} // namespace sxt::sqcgn
