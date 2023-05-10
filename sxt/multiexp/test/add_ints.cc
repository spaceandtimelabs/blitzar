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
#include "sxt/multiexp/test/add_ints.h"

#include <cstdlib>
#include <iostream>

#include "sxt/base/error/panic.h"

namespace sxt::mtxtst {
//--------------------------------------------------------------------------------------------------
// add_ints
//--------------------------------------------------------------------------------------------------
void add_ints(basct::span<uint64_t> result, basct::cspan<basct::cspan<uint64_t>> terms,
              basct::cspan<uint64_t> inputs) noexcept {
  if (result.size() != terms.size()) {
    baser::panic("result.size() != terms.size()");
  }
  for (size_t result_index = 0; result_index < result.size(); ++result_index) {
    auto& res_i = result[result_index];
    res_i = 0;
    for (auto term_index : terms[result_index]) {
      if (term_index >= inputs.size()) {
        baser::panic("term_index >= inputs.size()");
      }
      res_i += inputs[term_index];
    }
  }
}
} // namespace sxt::mtxtst
