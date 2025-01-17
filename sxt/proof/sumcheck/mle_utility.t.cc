/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2025-present Space and Time Labs, Inc.
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
#include "sxt/proof/sumcheck/mle_utility.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/scalar25/type/element.h"

using namespace sxt;
using namespace sxt::prfsk;

TEST_CASE("we can query the fraction of device memory taken by MLEs") {
  std::vector<s25t::element> mles;

  SECTION("we handle the zero case") { REQUIRE(get_gpu_memory_fraction(mles) == 0.0); }

  SECTION("the fractions doubles if the length of mles doubles") {
    mles.resize(1);
    auto f1 = get_gpu_memory_fraction(mles);
    REQUIRE(f1 > 0);
    mles.resize(2);
    auto f2 = get_gpu_memory_fraction(mles);
    REQUIRE(f2 == Catch::Approx(2 * f1));
  }
}
