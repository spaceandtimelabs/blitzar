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
#include "benchmark/multi_exp1/multi_exp_cpu.h"

#include "benchmark/multi_exp1/multiply_add.h"
#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/type/element_p3.h"

namespace sxt {
//--------------------------------------------------------------------------------------------------
// multi_exp_cpu
//--------------------------------------------------------------------------------------------------
void multi_exp_cpu(c21t::element_p3* res, int m, int n) noexcept {
  for (int mi = 0; mi < m; ++mi) {
    auto& res_mi = res[mi];
    res_mi = c21cn::zero_p3_v;
    for (int i = 0; i < n; ++i) {
      multiply_add(res_mi, mi, i);
    }
  }
}
} // namespace sxt
