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
#include "sxt/seqcommit/generator/precomputed_generators.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/seqcommit/generator/base_element.h"

using namespace sxt;
using namespace sxt::sqcgn;

TEST_CASE("we can precompute generators") {
  // we start with no generators precomputed
  auto generators = get_precomputed_generators();
  REQUIRE(generators.empty());

  // if we precompute generators, we can access them
  init_precomputed_generators(10, false);
  generators = get_precomputed_generators();
  REQUIRE(generators.size() == 10);

  // the precomputed generators match the computed values
  c21t::element_p3 e;
  compute_base_element(e, 0);
  REQUIRE(generators[0] == e);

  compute_base_element(e, 9);
  REQUIRE(generators[9] == e);

  std::vector<c21t::element_p3> data;
  generators = get_precomputed_generators(data, 10, 0, false);
  REQUIRE(data.empty());
  compute_base_element(e, 9);
  REQUIRE(generators[9] == e);

  // we get correct generators when `data.length() > precomputed.length()`
  generators = get_precomputed_generators(data, 12, 0, false);
  REQUIRE(data.size() == 12);
  compute_base_element(e, 11);
  REQUIRE(generators[11] == e);
  REQUIRE(generators.size() == 12);

  // we get correct generators when `offset != 0` and
  // `offset + data.length() <= precomputed.length()`
  generators = get_precomputed_generators(data, 2, 4, false);
  REQUIRE(data.size() == 12); // data should not be modified
  compute_base_element(e, 4);
  REQUIRE(generators[0] == e);
  REQUIRE(generators.size() == 2);

  generators = get_precomputed_generators(data, 6, 4, false);
  REQUIRE(data.size() == 12); // data should not be modified
  compute_base_element(e, 4);
  REQUIRE(generators[0] == e);
  compute_base_element(e, 9);
  REQUIRE(generators[5] == e);
  REQUIRE(generators.size() == 6);

  // we get correct generators when `offset != 0` and `offset < precomputed.length()`,
  // but `offset + data.length() > precomputed.length()`
  generators = get_precomputed_generators(data, 8, 3, false);
  REQUIRE(data.size() == 8);
  compute_base_element(e, 3);
  REQUIRE(generators[0] == e);
  compute_base_element(e, 10);
  REQUIRE(generators[7] == e);
  REQUIRE(generators.size() == 8);

  // we get correct generators when `offset > precomputed.length()`
  generators = get_precomputed_generators(data, 2, 12, false);
  REQUIRE(data.size() == 2);
  compute_base_element(e, 12);
  REQUIRE(generators[0] == e);
  compute_base_element(e, 13);
  REQUIRE(generators[1] == e);
  REQUIRE(generators.size() == 2);
}
