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
#include "sxt/seqcommit/test/test_generators.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/seqcommit/generator/base_element.h"

namespace sxt::sqctst {
//--------------------------------------------------------------------------------------------------
// test_pedersen_get_generators
//--------------------------------------------------------------------------------------------------
void test_pedersen_get_generators(
    basf::function_ref<void(basct::span<c21t::element_p3> generators, uint64_t offset_generators)>
        f) {

  SECTION("we can verify that computed generators are correct when offset is zero") {
    c21t::element_p3 expected_g_0, expected_g_1;
    uint64_t num_generators = 2;
    uint64_t offset_generators = 0;

    sqcgn::compute_base_element(expected_g_0, 0 + offset_generators);
    sqcgn::compute_base_element(expected_g_1, 1 + offset_generators);

    c21t::element_p3 generators[num_generators];
    basct::span<c21t::element_p3> span_generators(generators, num_generators);

    f(span_generators, offset_generators);

    REQUIRE(generators[0] == expected_g_0);
    REQUIRE(generators[1] == expected_g_1);
  }

  SECTION("we can verify that computed generators are correct when offset is non zero") {
    c21t::element_p3 expected_g_0, expected_g_1;
    uint64_t num_generators = 2;
    uint64_t offset_generators = 15;

    sqcgn::compute_base_element(expected_g_0, 0 + offset_generators);
    sqcgn::compute_base_element(expected_g_1, 1 + offset_generators);

    c21t::element_p3 generators[num_generators];
    basct::span<c21t::element_p3> span_generators(generators, num_generators);

    f(span_generators, offset_generators);

    REQUIRE(generators[0] == expected_g_0);
    REQUIRE(generators[1] == expected_g_1);
  }
}
} // namespace sxt::sqctst
