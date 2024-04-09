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
#include "sxt/field32/operation/invert.h"

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/field32/constant/one.h"
#include "sxt/field32/operation/mul.h"
#include "sxt/field32/random/element.h"
#include "sxt/field32/type/element.h"

using namespace sxt;
using namespace sxt::f32o;

TEST_CASE("inversion") {
  SECTION("of a random field element multiplied by its inverse is equal to one") {
    f32t::element a;
    basn::fast_random_number_generator rng{1, 2};
    f32rn::generate_random_element(a, rng);

    f32t::element a_inv;
    auto is_zero = invert(a_inv, a);
    REQUIRE(!is_zero);

    f32t::element ret_mul;
    mul(ret_mul, a, a_inv);
    REQUIRE(ret_mul == f32cn::one_v);
  }
}
