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
#include "sxt/proof/inner_product/generator_fold_kernel.h"

#include <vector>

#include "sxt/base/test/unit_test.h"
#include "sxt/curve32/operation/overload.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/proof/inner_product/generator_fold.h"
#include "sxt/ristretto/operation/compression.h"
#include "sxt/ristretto/type/literal.h"
#include "sxt/scalar25/constant/max_bits.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfip;

using s25t::operator""_s25;
using rstt::operator""_rs;

TEST_CASE("we can fold generators using the GPU") {
  unsigned data[s25cn::max_bits_v];
  basct::span<unsigned> decomposition{data};

  auto x1 = 0x123_s25;
  auto x2 = 0x321_s25;
  decompose_generator_fold(decomposition, x1, x2);

  std::vector<c32t::element_p3> g_vector_p;
  std::vector<c32t::element_p3> g_vector;

  rstt::compressed_element expected, actual;

  SECTION("we can fold two generators") {
    g_vector = {0x112233_rs, 0x332211_rs};
    g_vector_p.resize(1);

    auto fut = async_fold_generators(g_vector_p, g_vector, decomposition);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());
    rsto::compress(expected, x1 * g_vector[0] + x2 * g_vector[1]);
    rsto::compress(actual, g_vector_p[0]);
    REQUIRE(actual == expected);
  }

  SECTION("we can fold four generators") {
    g_vector = {0x11_rs, 0x22_rs, 0x33_rs, 0x44_rs};
    g_vector_p.resize(2);

    auto fut = async_fold_generators(g_vector_p, g_vector, decomposition);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());

    rsto::compress(expected, x1 * g_vector[0] + x2 * g_vector[2]);
    rsto::compress(actual, g_vector_p[0]);
    REQUIRE(actual == expected);

    rsto::compress(expected, x1 * g_vector[1] + x2 * g_vector[3]);
    rsto::compress(actual, g_vector_p[1]);
    REQUIRE(actual == expected);
  }
}
