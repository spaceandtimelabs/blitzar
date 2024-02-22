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
#include "sxt/field32/constant/d.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/field32/type/literal.h"

using namespace sxt::f32cn;
using namespace sxt::f32t;

TEST_CASE("constants") {
  SECTION("d_v") {
    REQUIRE(d_v == 0x52036cee2b6ffe738cc740797779e89800700a4d4141d8ab75eb4dca135978a3_f32);
  }

  SECTION("d2_v") {
    REQUIRE(d2_v == 0x2406d9dc56dffce7198e80f2eef3d13000e0149a8283b156ebd69b9426b2f159_f32);
  }

  SECTION("onemsqd_v") {
    REQUIRE(onemsqd_v == 0x29072a8b2b3e0d79994abddbe70dfe42c81a138cd5e350fe27c09c1945fc176_f32);
  }

  SECTION("sqdmone_v") {
    REQUIRE(sqdmone_v == 0x5968b37af66c22414cdcd32f529b4eebd29e4a2cb01e199931ad5aaa44ed4d20_f32);
  }

  SECTION("sqdmone_v") {
    REQUIRE(sqdmone_v == 0x5968b37af66c22414cdcd32f529b4eebd29e4a2cb01e199931ad5aaa44ed4d20_f32);
  }

  SECTION("sqrtadm1_v") {
    REQUIRE(sqrtadm1_v == 0x376931bf2b8348ac0f3cfcc931f5d1fdaf9d8e0c1b7854bd7e97f6a0497b2e1b_f32);
  }
}
