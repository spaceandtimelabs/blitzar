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
#include "sxt/field32/operation/pow22523.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field32/type/element.h"
#include "sxt/field32/type/literal.h"

using namespace sxt;
using namespace sxt::f32o;
using namespace sxt::f32t;

TEST_CASE("pow22523") {
  SECTION("first test") {
    auto a = 0x5187889dbf41151c5858a2cc5924e0a9e34592749088b9866d452769294eca83_f32;
    f32t::element ret;
    pow22523(ret, a);
    REQUIRE(ret == 0x47cf91da87b2194cbf5d55714e7374c34695fb7cd921a7d46e3a21fd801c59df_f32);
  }

  SECTION("second test") {
    auto a = 0x7397e3032f73c74bb068e931363dc16ea16bb74d68e16d331ca2f087c70eef70_f32;
    f32t::element ret;
    pow22523(ret, a);
    REQUIRE(ret == 0x3b09e81f10a8e1e0862e16e2587b824ee82bbb72c9efdd713df68f2eb0af79ea_f32);
  }
}
