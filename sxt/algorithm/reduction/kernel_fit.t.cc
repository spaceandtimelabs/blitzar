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
#include "sxt/algorithm/reduction/kernel_fit.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/execution/kernel/kernel_dims.h"

using namespace sxt;
using namespace sxt::algr;

TEST_CASE("we can determine the dimensions of a reduction kernel") {
  SECTION("we handle the smallest case") {
    auto dims = fit_reduction_kernel(1);
    REQUIRE(dims.num_blocks == 1);
    REQUIRE(dims.block_size == xenk::block_size_t::v1);
  }

  SECTION("for reductions on the device, all threads have work to do") {
    for (unsigned int n : {64, 65, 127, 128, 100'000, 1'000'000}) {
      auto dims = fit_reduction_kernel(n);
      REQUIRE(dims.num_blocks > 0);
      REQUIRE(dims.num_blocks * static_cast<unsigned int>(dims.block_size) * 2 <= n);
    }
  }
}
