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
#include "sxt/seqcommit/generator/precomputed_one_commitments.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/seqcommit/generator/cpu_one_commitments.h"

using namespace sxt;
using namespace sxt::sqcgn;

TEST_CASE("we can precompute one commitments") {
  // we start with no generators precomputed
  auto one_commitments = get_precomputed_one_commitments();
  REQUIRE(one_commitments.empty());
  REQUIRE(get_precomputed_one_commit(0) == cpu_get_one_commit(0));

  // if we precompute generators, we can access them
  init_precomputed_one_commitments(10);

  one_commitments = get_precomputed_one_commitments();

  REQUIRE(one_commitments.size() == 10);
  REQUIRE(one_commitments[0] == cpu_get_one_commit(0));
  REQUIRE(one_commitments[9] == cpu_get_one_commit(9));
  REQUIRE(get_precomputed_one_commit(0) == cpu_get_one_commit(0));
  REQUIRE(get_precomputed_one_commit(9) == cpu_get_one_commit(9));
  REQUIRE(get_precomputed_one_commit(10) == cpu_get_one_commit(10));
  REQUIRE(get_precomputed_one_commit(15) == cpu_get_one_commit(15));
}
