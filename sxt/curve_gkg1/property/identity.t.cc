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
#include "sxt/curve_gkg1/property/identity.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/curve_gkg1/constant/generator.h"
#include "sxt/curve_gkg1/type/element_affine.h"
#include "sxt/curve_gkg1/type/element_p2.h"

using namespace sxt;
using namespace sxt::ck1p;

TEST_CASE("the identity can be identified") {
  SECTION("as an affine element") {
    REQUIRE(is_identity(ck1t::element_affine::identity()));
    REQUIRE(!is_identity(ck1cn::generator_affine_v));
  }

  SECTION("as a projective element") {
    REQUIRE(is_identity(ck1t::element_p2::identity()));
    REQUIRE(!is_identity(ck1cn::generator_p2_v));
  }
}
