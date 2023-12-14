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
#include "sxt/field_mtg/operation/add.h"

#include "sxt/base/field/example_element.h"
#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::fmtgo;

TEST_CASE("addition") {
  SECTION("works on basic elements") {
    constexpr basfld::element1 a{6};
    constexpr basfld::element1 b{14};
    constexpr basfld::element1 exp{20};

    basfld::element1 ret;
    add(ret, a, b);

    REQUIRE(ret == exp);
  }

  SECTION("respects the modulus") {
    constexpr basfld::element1 a{90};
    constexpr basfld::element1 b{10};
    constexpr basfld::element1 exp{3};

    basfld::element1 ret;
    add(ret, a, b);

    REQUIRE(ret == exp);
  }
}
