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
#include "sxt/curve_g1/operation/mul_by_3b.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field12/constant/one.h"
#include "sxt/field12/type/element.h"
#include "sxt/field12/type/literal.h"

using namespace sxt;
using namespace sxt::cg1o;
using namespace sxt::f12t;

TEST_CASE("multiply by 3b") {
  SECTION("returns twelve if one in Montogomery form is the input") {
    f12t::element ret;

    mul_by_3b(ret, f12cn::one_v);

    REQUIRE(0xc_f12 == ret);
  }
}
