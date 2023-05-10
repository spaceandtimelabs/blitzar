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
#include "sxt/algorithm/iteration/kernel_fit.h"

#include <iostream>

#include "sxt/base/num/divide_up.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/kernel/kernel_dims.h"

using namespace sxt;
using namespace sxt::algi;

TEST_CASE("we determine valid kernel dimensions for iterations") {
  for (unsigned n : {1, 2, 3, 4, 5, 31, 32, 33, 63, 64, 100, 1'000, 10'000, 100'000, 1'000'000}) {
    auto dims = fit_iteration_kernel(n);
    auto num_iters = basn::divide_up(n, static_cast<unsigned>(dims.block_size) * dims.num_blocks) *
                     static_cast<unsigned>(dims.block_size);
    REQUIRE(dims.num_blocks > 0);
    REQUIRE((dims.num_blocks - 1) * num_iters < n);
    REQUIRE(n <= dims.num_blocks * num_iters);
  }
}
