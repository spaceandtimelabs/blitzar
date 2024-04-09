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
#include "sxt/curve_bng1_32/operation/cmov.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve_bng1_32/constant/generator.h"
#include "sxt/curve_bng1_32/type/element_affine.h"
#include "sxt/curve_bng1_32/type/element_p2.h"

using namespace sxt;
using namespace sxt::cn3o;

TEST_CASE("cmov returns the expected projective coordinates") {
  cn3t::element_p2 expect_generator{cn3cn::generator_p2_v};
  cn3t::element_p2 expect_identity{cn3t::element_p2::identity()};

  cn3o::cmov(expect_generator, cn3t::element_p2::identity(), 0);
  cn3o::cmov(expect_identity, cn3t::element_p2::identity(), 1);

  REQUIRE(expect_generator == cn3cn::generator_p2_v);
  REQUIRE(expect_identity == cn3t::element_p2::identity());
}
