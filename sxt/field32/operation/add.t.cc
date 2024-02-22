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
#include "sxt/field32/operation/add.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field32/type/element.h"
#include "sxt/field32/type/literal.h"

using namespace sxt;
using namespace sxt::f32o;
using namespace sxt::f32t;

TEST_CASE("add") {
  SECTION("with elements") {
    f32t::element a{12775011, 23676731, 59192183, 16867033, 23768340, 1913951, 6461676, 11632506, 6764333, 17261081};
    f32t::element b{58368107, 31446661, 49405094, 30328970, 25558377, 26518888, 65879413, 25699118, 56941748, 10839393};
    f32t::element expected{4034254, 21568961, 41488414, 13641572, 49326718, 28432839, 5232225, 3777193, 63706082, 28100474};

    f32t::element ret;
    add(ret, a, b);
    REQUIRE(ret == expected);
  }

  SECTION("with literal") {
    auto a = 0x5187889dbf41151c5858a2cc5924e0a9e34592749088b9866d452769294eca83_f32;
    auto b = 0x7397e3032f73c74bb068e931363dc16ea16bb74d68e16d331ca2f087c70eef70_f32;
    auto expected_sum = 0x451f6ba0eeb4dc6808c18bfd8f62a21884b149c1f96a26b989e817f0f05dba06_f32;

    f32t::element sum;

    add(sum, a, b);

    REQUIRE(sum == expected_sum);
  }
}
