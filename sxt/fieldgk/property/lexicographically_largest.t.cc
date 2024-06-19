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
#include "sxt/fieldgk/property/lexicographically_largest.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/fieldgk/base/montgomery.h"
#include "sxt/fieldgk/constant/one.h"
#include "sxt/fieldgk/constant/zero.h"
#include "sxt/fieldgk/type/element.h"
#include "sxt/fieldgk/type/literal.h"

using namespace sxt;
using namespace sxt::fgkp;
using namespace sxt::fgkt;

TEST_CASE("lexicographically largest correctly identifies") {
  SECTION("zero is not largest") { REQUIRE(!lexicographically_largest(fgkcn::zero_v)); }

  SECTION("one is not largest") { REQUIRE(!lexicographically_largest(fgkcn::one_v)); }

  SECTION("(p_v-1)/2 in Montgomery form is not largest") {
    element e = 0x183227397098d014dc2822db40c0ac2e9419f4243cdcb848a1f0fac9f8000000_fgk;
    element e_montgomery;
    fgkb::to_montgomery_form(e_montgomery.data(), e.data());

    REQUIRE(!lexicographically_largest(e_montgomery));
  }

  SECTION("((p_v-1)/2)+1 in Montgomery form is largest") {
    element e = 0x183227397098d014dc2822db40c0ac2e9419f4243cdcb848a1f0fac9f8000001_fgk;
    element e_montgomery;
    fgkb::to_montgomery_form(e_montgomery.data(), e.data());

    REQUIRE(lexicographically_largest(e_montgomery));
  }
}
