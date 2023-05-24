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
#include "sxt/seqcommit/generator/precomputed_initializer.h"

#include "sxt/seqcommit/generator/precomputed_generators.h"
#include "sxt/seqcommit/generator/precomputed_one_commitments.h"

namespace sxt::sqcgn {
//--------------------------------------------------------------------------------------------------
// init_precomputed_components
//--------------------------------------------------------------------------------------------------
void init_precomputed_components(size_t n, bool use_gpu) noexcept {
  // generators must be initialized before one_commitments as the latter uses the first
  init_precomputed_generators(n, use_gpu);
  init_precomputed_one_commitments(n);
}
} // namespace sxt::sqcgn
