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

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/seqcommit/generator/cpu_generator.h"

using namespace sxt;
using namespace sxt::sqcgn;

TEST_CASE("test one commitments") {
  std::vector<c21t::element_p3> precomputed_values(5), generators(5);

  cpu_get_generators(generators, 0);
  cpu_get_one_commitments(precomputed_values);

  c21t::element_p3 sum_gen_0_1;
  c21o::add(sum_gen_0_1, generators[0], generators[1]);

  SECTION("we can correctly generate precomputed values") {
    REQUIRE(c21t::element_p3::identity() == precomputed_values[0]);
    REQUIRE(generators[0] == precomputed_values[1]);
    REQUIRE(sum_gen_0_1 == precomputed_values[2]);
  }

  // we can compute `cpu_get_one_commit`
  SECTION("we can correctly generate one commitments at i-th position out of identity value") {
    REQUIRE(c21t::element_p3::identity() == cpu_get_one_commit(0));
    REQUIRE(generators[0] == cpu_get_one_commit(1));
    REQUIRE(sum_gen_0_1 == cpu_get_one_commit(2));
  }

  // we can compute `cpu_get_one_commit` with predefined commitments and offsets
  SECTION("we can correctly generate one commitments at i-th position out of a predefined "
          "commitment and offset") {
    REQUIRE(c21t::element_p3::identity() == cpu_get_one_commit(c21t::element_p3::identity(), 0, 0));
    REQUIRE(c21t::element_p3::identity() == cpu_get_one_commit(c21t::element_p3::identity(), 0, 1));
    REQUIRE(generators[0] == cpu_get_one_commit(c21t::element_p3::identity(), 1, 0));
    REQUIRE(generators[0] == cpu_get_one_commit(generators[0], 0, 0));
    REQUIRE(generators[0] == cpu_get_one_commit(generators[0], 0, 1));
    REQUIRE(sum_gen_0_1 == cpu_get_one_commit(c21t::element_p3::identity(), 2, 0));
    REQUIRE(sum_gen_0_1 == cpu_get_one_commit(generators[0], 1, 1));
  }
}
